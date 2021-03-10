# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Solves linear optimal dispatch in hourly resolution
using the capacities of previous capacity expansion in rule :mod:`solve_network`.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
        solver:
            name:
            (solveroptions):

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`solving_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`
- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`solve`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op.nc``: Solved PyPSA network for optimal dispatch including optimisation results

Description
-----------

"""

import logging
from _helpers import configure_logging

import pypsa
import numpy as np

from pathlib import Path
from vresutils.benchmark import memory_logger
from solve_network import solve_network, prepare_network
from build_optimized_capacities_iteration1 import calculate_nodal_capacities
from six import iteritems
import pandas as pd
import os
from solve_network import solve_network, prepare_network, patch_pyomo_tmpdir

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

def change_co2limit(n, Nyears=1., factor=None):
    if factor is not None:
        annual_emissions = factor*snakemake.config['electricity']['co2base']
    else:
        annual_emissions = snakemake.config['electricity']['co2limit']
    n.global_constraints.loc["CO2Limit", "constant"] = annual_emissions * Nyears

def set_parameters_from_optimized(n, networks_dict, solve_opts):
    nodal_capacities = calculate_nodal_capacities(networks_dict)

    links_dc_i = n.links.index[n.links.p_nom_extendable]
    links_capacities = nodal_capacities.loc['links']
    n.links.loc[links_dc_i, 'p_nom'] = links_capacities.loc[links_dc_i,:].max(axis=1)
    n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    gen_capacities = nodal_capacities.loc['generators']
   # gen_extend_i_exclude_biomass = [elem for i, elem in enumerate(gen_extend_i) if elem not in biomass_extend_index]
    n.generators.loc[gen_extend_i, 'p_nom'] = gen_capacities.loc[gen_extend_i,:].max(axis=1)
    n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False
    extra_generator = solve_opts.get('extra_generator')
    if extra_generator in snakemake.config["electricity"]["conventional_carriers"]:
        if extra_generator == "OCGT":
            change_co2limit(n, 1, 0.05)
        generator_extend_index = n.generators.index[n.generators.carrier == extra_generator]
        n.generators.loc[generator_extend_index, 'p_nom'] = gen_capacities.loc[generator_extend_index, :].max(axis=1)
        n.generators.loc[generator_extend_index, 'p_nom_extendable'] = False

    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    stor_capacities = nodal_capacities.loc['storage_units']
    n.storage_units.loc[stor_extend_i, 'p_nom'] = stor_capacities.loc[stor_extend_i, :].max(axis=1)
    n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    stores_capacities = nodal_capacities.loc['stores']
    n.stores.loc[stores_extend_i, 'e_nom'] = stores_capacities.loc[stores_extend_i, :].max(axis=1)
    n.stores.loc[stores_extend_i, 'e_nom_extendable'] = False
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
        os.path.join(network_dir, 'iteration4', f'{network}_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}_{planning_horizons}.ncc')
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