

import logging
from _helpers import configure_logging

import pypsa
import numpy as np

from pathlib import Path
from vresutils.benchmark import memory_logger
from solve_network import solve_network, prepare_network
from build_optimized_capacities_iteration1 import calculate_nodal_capacities
from add_electricity import load_costs, load_powerplants, attach_conventional_generators, _add_missing_carriers_from_costs
from solve_network import solve_network, prepare_network, patch_pyomo_tmpdir
from six import iteritems
import pandas as pd
import os

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}

def add_extra_generator(n, solve_opts):
    extra_generator = solve_opts.get('extra_generator')

    if extra_generator == 'load_shedding':
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               #sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e6, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
               )
    else:
        Nyears = n.snapshot_weightings.sum() / 8760.
        costs = "data/costs.csv"
        costs = load_costs(Nyears, tech_costs = costs, config = snakemake.config['costs'], elec_config = snakemake.config['electricity'])
        ppl = load_powerplants(ppl_fn='resources/powerplants.csv')
        carriers = extra_generator

        _add_missing_carriers_from_costs(n, costs, carriers)

        ppl = (ppl.query('carrier in @carriers').join(costs, on='carrier')
               .rename(index=lambda s: 'C' + str(s)))

        logger.info('Adding {} generators with capacities [MW] \n{}'
                    .format(len(ppl), ppl.groupby('carrier').p_nom.sum()))

        n.madd("Generator", ppl.index,
               carrier=ppl.carrier,
               bus=ppl.bus,
               p_nom=ppl.p_nom,
               efficiency=ppl.efficiency,
               marginal_cost=ppl.marginal_cost,
               capital_cost=0)

        logger.warning(f'Capital costs for conventional generators put to 0 EUR/MW.')

    return n

def change_co2limit(n, Nyears=1., factor=None):
    if factor is not None:
        annual_emissions = factor*snakemake.config['electricity']['co2base']
    else:
        annual_emissions = snakemake.config['electricity']['co2limit']
    n.global_constraints.loc["CO2Limit", "constant"] = annual_emissions * Nyears

def set_parameters_from_optimized(n, networks_dict, solve_opts):
    nodal_capacities = calculate_nodal_capacities(networks_dict)

    # lines_typed_i = n.lines.index[n.lines.type != '']
    # n.lines.loc[lines_typed_i, 'num_parallel'] = \
    #     n_optim.lines['num_parallel'].reindex(lines_typed_i, fill_value=0.)
    # n.lines.loc[lines_typed_i, 's_nom'] = (
    #     np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
    #     n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel)
    #
    # lines_untyped_i = n.lines.index[n.lines.type == '']
    # for attr in ('s_nom', 'r', 'x'):
    #     n.lines.loc[lines_untyped_i, attr] = n_optim.lines[attr].reindex(lines_untyped_i, fill_value=0.)
    # n.lines['s_nom_extendable'] = False

    # lines_extend_i = n.lines.index[n.lines.s_nom_extendable]
    # lines_capacities = nodal_capacities.loc['lines']
    # print(lines_capacities)
    # lines_capacities = lines_capacities.reset_index(level=[1]).reindex(lines_extend_i, fill_value=0.)
    # n.lines.loc[lines_extend_i, 's_nom_max'] = lines_capacities.loc[lines_extend_i,:].max(axis=1)
    # n.lines.loc[lines_extend_i, 's_nom_min'] = lines_capacities.loc[lines_extend_i, :].mean(axis=1)
    # n.lines.loc[lines_extend_i, 's_nom_extendable'] = False

    links_dc_i = n.links.index[n.links.p_nom_extendable]
    links_capacities = nodal_capacities.loc['links']
    n.links.loc[links_dc_i, 'p_nom'] = links_capacities.loc[links_dc_i,:].mean(axis=1)
    n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    #
    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    gen_capacities = nodal_capacities.loc['generators']
   # gen_extend_i_exclude_biomass = [elem for i, elem in enumerate(gen_extend_i) if elem not in biomass_extend_index]
    n.generators.loc[gen_extend_i, 'p_nom'] = gen_capacities.loc[gen_extend_i,:].mean(axis=1)
    n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False
    extra_generator = solve_opts.get('extra_generator')
    print("extra_generator")
    print(extra_generator)
    print(snakemake.config["electricity"]["conventional_carriers"])
    if extra_generator in snakemake.config["electricity"]["conventional_carriers"]:
        if extra_generator == "OCGT":
            change_co2limit(n, 1, 0.05)
        print("here1")
        generator_extend_index = n.generators.index[n.generators.carrier == extra_generator]
        n.generators.loc[generator_extend_index, 'p_nom_extendable'] = True
        print(generator_extend_index)
        print("here")


    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    stor_capacities = nodal_capacities.loc['storage_units']
    n.storage_units.loc[stor_extend_i, 'p_nom'] = stor_capacities.loc[stor_extend_i, :].mean(axis=1)
    n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    stores_capacities = nodal_capacities.loc['stores']
    n.stores.loc[stores_extend_i, 'e_nom'] = stores_capacities.loc[stores_extend_i, :].mean(axis=1)
    n.stores.loc[stores_extend_i, 'e_nom_extendable']\

    return n

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict

        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='40', lv='1.0',
                           sector_opts='Co2L0p0-3H-T-H-B-I',
                           co2_budget_name='b30b3', planning_horizons='2030'),
            input=dict(
                network="pypsa-eur-sec/results/test/prenetworks_brownfield/{network}_s{simpl}_{clusters}_lv{lv}__{sector_opts}_{co2_budget_name}_{planning_horizons}.nc"),
            output=[
                "results/networks/s{simpl}_{clusters}_lv{lv}_{sector_opts}_{co2_budget_name}_{planning_horizons}-test.nc"],
            log=dict(
                gurobi="logs/{network}_s{simpl}_{clusters}_lv{lv}_{sector_opts}_{co2_budget_name}_{planning_horizons}_gurobi-test.log",
                python="logs/{network}_s{simpl}_{clusters}_lv{lv}_{sector_opts}_{co2_budget_name}_{planning_horizons}_python-test.log")
        )
        import yaml

        with open('config.yaml', encoding='utf8') as f:
            snakemake.config = yaml.safe_load(f)
        network_dir = os.path.join('..', 'results', 'networks')
    else:
        network_dir = os.path.join('results', 'networks')
    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        patch_pyomo_tmpdir(tmpdir)

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        n = pypsa.Network(snakemake.input.network,
                          override_component_attrs=override_component_attrs)


    def expand_from_wildcard(key):
        w = getattr(snakemake.wildcards, key)
        return snakemake.config["scenario"][key] if w == "all" else [w]


    # refered = pathlib.Path(snakemake.input.network_opti)

    networks_dict = {(capacity_year) :
        os.path.join(network_dir, 'iteration3', f'{network}_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}_{planning_horizons}.ncc')
                     for capacity_year in snakemake.config["scenario"]["capacity_years"]
                     for network in expand_from_wildcard("network")
                     for simpl in expand_from_wildcard("simpl")
                     for clusters in expand_from_wildcard("clusters")
                     for lv in lv
                     for opts in expand_from_wildcard("opts")
                     for sector_opts in expand_from_wildcard("sector_opts")
                     for planning_horizons in expand_from_wildcard("planning_horizons")}
    print(networks_dict)

    # n_optim = pypsa.Network(snakemake.input.network_opti,
    #                         override_component_attrs=override_component_attrs)
    n = prepare_network(n)
    n = set_parameters_from_optimized(n, networks_dict)



    n = solve_network(n)
    n.export_to_netcdf(snakemake.output[0])