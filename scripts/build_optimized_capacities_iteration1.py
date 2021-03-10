import sys

sys.path = ["/home/vres/data/tom/lib/pypsa"] + sys.path

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
import gc
import os

import pypsa

from pypsa.descriptors import free_output_series_dataframes
from six import iteritems
from solve_network import solve_network, prepare_network, patch_pyomo_tmpdir


opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}
# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger

# First tell PyPSA that links can have multiple outputs by
# overriding the component_attrs. This can be done for
# as many buses as you need with format busi for i = 2,3,4,5,....
# See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs


override_component_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string", np.nan, np.nan, "2nd bus", "Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string", np.nan, np.nan, "3rd bus", "Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series", "per unit", 1., "2nd bus efficiency",
                                                       "Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series", "per unit", 1., "3rd bus efficiency",
                                                       "Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series", "MW", 0., "2nd bus output", "Output"]
override_component_attrs["Link"].loc["p3"] = ["series", "MW", 0., "3rd bus output", "Output"]




def patch_pyomo_tmpdir(tmpdir):
    # PYOMO should write its lp files into tmp here
    import os
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    from pyutilib.services import TempfileManager
    TempfileManager.tempdir = tmpdir

def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ",start=4),c.df.index)
        for i in ifind.unique():
            names = ifind.index[ifind == i]
            if i == -1:
                c.df.loc[names,'location'] = ""
            else:
                c.df.loc[names,'location'] = names.str[:i]

def calculate_nodal_capacities(networks_dict):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    # networks_dict = {(year):'results/networks/{year}/elec_s_40_ec_lv1.0_Co2L0p0-3H_storage_units.nc' \
    #                          .format(year=year) for year in capacity_years}
    columns = list(networks_dict.keys())
    nodal_capacities = pd.DataFrame(columns=columns)
    lines_capacities = pd.DataFrame()

    for label, filename in iteritems(networks_dict):
        if not os.path.exists(filename):
            continue
        n = pypsa.Network(filename)
        assign_carriers(n)
        assign_locations(n)

        for c in n.iterate_components(n.branch_components ^ {"Transformer"}| n.controllable_one_port_components ^ {"Load"}):
            #nodal_capacities_c = c.df.groupby(["location", "carrier"])[opt_name.get(c.name, "p") + "_nom_opt"].sum()
            nodal_capacities_c = c.df[opt_name.get(c.name, "p") + "_nom_opt"]
           # print([(c.list_name,) + t for t in nodal_capacities_c.index])
            index = pd.MultiIndex.from_tuples([(c.list_name,t) for t in nodal_capacities_c.index])
            nodal_capacities = nodal_capacities.reindex(index | nodal_capacities.index)
            nodal_capacities.loc[index, label] = nodal_capacities_c.values
        # df = pd.concat([nodal_capacities, df], axis=1)
    nodal_capacities.to_csv("notebook/data/nodal_capacities.csv")
    return nodal_capacities

def set_parameters_from_optimized(n, networks_dict):
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
    n.links.loc[links_dc_i, 'p_nom_max'] = links_capacities.loc[links_dc_i,:].max(axis=1)
    n.links.loc[links_dc_i, 'p_nom_min'] = links_capacities.loc[links_dc_i,:].mean(axis=1)
   # n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    gen_capacities = nodal_capacities.loc['generators']
    #gen_capacities = gen_capacities.reset_index(level=[1]).reindex(gen_extend_i, fill_value=0.)
    n.generators.loc[gen_extend_i, 'p_nom_max'] = gen_capacities.loc[gen_extend_i,:].max(axis=1)
    n.generators.loc[gen_extend_i, 'p_nom_min'] = gen_capacities.loc[gen_extend_i,:].mean(axis=1)
   # n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False

    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    stor_capacities = nodal_capacities.loc['storage_units']
    n.storage_units.loc[stor_extend_i, 'p_nom_max'] = stor_capacities.loc[stor_extend_i,:].max(axis=1)
    n.storage_units.loc[stor_extend_i, 'p_nom_min'] = stor_capacities.loc[stor_extend_i, :].mean(axis=1)
   # n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    stores_capacities = nodal_capacities.loc['stores']
    n.stores.loc[stores_extend_i, 'e_nom_max'] = stores_capacities.loc[stores_extend_i,:].max(axis=1)
    n.stores.loc[stores_extend_i, 'e_nom_min'] = stores_capacities.loc[stores_extend_i, :].mean(axis=1)
   # n.stores.loc[stores_extend_i, 'e_nom_extendable'] = False
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
        os.path.join(network_dir, 'iteration0', f'{network}_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}_{planning_horizons}.ncc')
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
