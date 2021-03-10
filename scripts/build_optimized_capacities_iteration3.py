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
from six import iteritems
from build_optimized_capacities_iteration1 import set_parameters_from_optimized
from solve_network import solve_network, prepare_network, patch_pyomo_tmpdir
import pandas as pd
import os

logger = logging.getLogger(__name__)

idx = pd.IndexSlice


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
        os.path.join(network_dir, 'iteration2', f'{network}_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}_{planning_horizons}.ncc')
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