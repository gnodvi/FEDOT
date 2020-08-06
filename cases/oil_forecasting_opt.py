import datetime
import os
from copy import copy
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from cases.oil_forecasting import get_comp_chain
from core.composer.chain import Chain
from core.composer.gp_composer.fixed_structure_composer import GPComposer
from core.composer.gp_composer.gp_composer import GPComposerRequirements
from core.composer.node import PrimaryNode
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root

seed(42)
np.random.seed(42)


def calculate_validation_metric(pred: OutputData, pred_crm, pred_crm_opt, valid: InputData,
                                name: str, is_visualise=False):
    forecast_length = pred.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[:, pred.predict.shape[1] - 1]
    predicted_crm = pred_crm.predict[:, pred_crm.predict.shape[1] - 1]
    predicted_crm_opt = pred_crm_opt.predict
    if len(predicted_crm_opt.shape) > 1:
        predicted_crm_opt = predicted_crm_opt[:, predicted_crm_opt.shape[1] - 1]

    real = valid.target[300:]
    real = real[:len(predicted)]

    predicted = predicted[4:]
    predicted_crm = predicted_crm[4:]
    predicted_crm_opt = predicted_crm_opt[4:]

    file_path_crm = f'D:/THEODOR/cases/data/oil/crimp.csv'

    data_frame = pd.read_csv(file_path_crm, sep=',')
    crm = data_frame[f'mean_{name}'][(300 - 4):700]
    crm[np.isnan(crm)] = 0
    crm = crm[:len(predicted)]
    # the quality assessment for the simulation results
    rmse_ml = mse(y_true=real, y_pred=predicted, squared=False)
    rmse_ml_crm = mse(y_true=real, y_pred=predicted_crm, squared=False)
    rmse_crm_opt = mse(y_true=real, y_pred=predicted_crm_opt, squared=False)
    rmse_crm = mse(y_true=real, y_pred=crm, squared=False)

    # plot results
    if is_visualise:
        compare_plot(predicted, predicted_crm, predicted_crm_opt, real,
                     forecast_length=forecast_length,
                     model_name=name, err=rmse_crm)
    return rmse_crm, rmse_ml, rmse_ml_crm, rmse_crm_opt


def compare_plot(predicted, predicted_crm, predicted_crm_opt, real, forecast_length, model_name, err):
    file_path_crm = f'D:/THEODOR/cases/data/oil/crimp.csv'

    data_frame = pd.read_csv(file_path_crm, sep=',')
    mean = data_frame[f'mean_{model_name}'][300:700]
    min_int = data_frame[f'min_{model_name}'][300:700]
    max_int = data_frame[f'max_{model_name}'][300:700]

    times = [_ for _ in range(len(mean))]

    plt.clf()
    _, ax = plt.subplots()
    # plt.figure(figsize=(10, 5))
    plt.plot(times, mean, label='CRM')
    plt.fill_between(times, min_int, max_int, alpha=0.2)
    plt.plot(real, linewidth=1, label="Observed", alpha=0.8)
    plt.plot(predicted, linewidth=1, label="ML", alpha=0.8)
    plt.plot(predicted_crm, linewidth=1, label="ML+CRM", alpha=0.8)
    plt.plot(predicted_crm_opt, linewidth=1, label="Evo ML+CRM", alpha=0.8)

    ax.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Oil volume')
    plt.title(f'Oil production for {forecast_length} hours in {model_name}, RMSE={round(err)} m3')
    plt.savefig(f'{model_name}.png')
    # plt.show()


def run_oil_forecasting_problem(train_file_path,
                                train_file_path_crm,
                                forecast_length, max_window_size,
                                is_visualise=False,
                                well_id='Unknown'):
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size,
                                             period=None))

    full_path_train = os.path.join(str(project_root()), train_file_path)
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    # a dataset for a final validation of the composed model
    full_path_test = os.path.join(str(project_root()), train_file_path)
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    full_path_train_crm = os.path.join(str(project_root()), train_file_path_crm)
    dataset_to_train_crm = InputData.from_csv(
        full_path_train_crm, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    dataset_to_validate_crm = dataset_to_train_crm

    prediction_full = None
    prediction_full_crm = None
    prediction_full_crm_opt = None

    composer = GPComposer()

    available_model_types = ['rfr', 'linear',
                             'ridge', 'lasso',
                             'knnreg', 'dtreg',
                             'treg', 'adareg',
                             'xgbreg']

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=4,
        max_depth=1, pop_size=10, num_of_generations=20,
        crossover_prob=0.6, mutation_prob=0.6,
        max_lead_time=datetime.timedelta(minutes=20),
        add_single_model_chains=True)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    node_1 = PrimaryNode('rfr')
    # node_2 = PrimaryNode('dtreg')
    # node_single = SecondaryNode('knnreg', nodes_from=[node_1, node_2])
    chain_template = get_comp_chain()  # Chain(node_1)

    for forecasting_step in range(4):

        if forecasting_step > 0:
            dataset_to_train_local_crm_prev = dataset_to_train_local_crm
            dataset_to_validate_local_crm_prev = dataset_to_validate_local_crm

        depth = 100
        start = 0 + depth * forecasting_step
        end = depth * 2 + depth * (forecasting_step + 1)
        dataset_to_train_local = copy(dataset_to_train)
        dataset_to_train_local.idx = dataset_to_train_local.idx[start:end]
        dataset_to_train_local.target = dataset_to_train_local.target[start:end]
        dataset_to_train_local.features = dataset_to_train_local.features[start:end, :]

        dataset_to_train_local_crm = copy(dataset_to_train_crm)
        dataset_to_train_local_crm.idx = dataset_to_train_local_crm.idx[start:end]
        dataset_to_train_local_crm.target = dataset_to_train_local_crm.target[start:end]
        dataset_to_train_local_crm.features = dataset_to_train_local_crm.features[start:end, :]

        # print(dataset_to_train_local.features.shape)

        start = 0 + depth * forecasting_step
        end = depth * 2 + depth * (forecasting_step + 1)

        dataset_to_validate_local = copy(dataset_to_validate)
        dataset_to_validate_local.idx = dataset_to_validate_local.idx[start + depth:end + depth]
        dataset_to_validate_local.target = dataset_to_validate_local.target[start + depth:end + depth]
        dataset_to_validate_local.features = dataset_to_validate_local.features[start + depth:end + depth, :]

        dataset_to_validate_local_crm = copy(dataset_to_validate_crm)
        dataset_to_validate_local_crm.idx = dataset_to_validate_local_crm.idx[start + depth:end + depth]
        dataset_to_validate_local_crm.target = dataset_to_validate_local_crm.target[start + depth:end + depth]
        dataset_to_validate_local_crm.features = dataset_to_validate_local_crm.features[start + depth:end + depth, :]

        # print(dataset_to_validate_local.features.shape)

        node_single = PrimaryNode('rfr')
        chain_simple = Chain(node_single)

        node_single2 = PrimaryNode('rfr')
        chain_simple2 = Chain(node_single2)

        chain_simple.fit(input_data=dataset_to_train_local, verbose=False)
        chain_simple2.fit(input_data=dataset_to_train_local_crm, verbose=False)

        prediction = chain_simple.predict(dataset_to_validate_local)
        prediction_crm = chain_simple2.predict(dataset_to_validate_local_crm)

        if forecasting_step > 0:
            dataset_to_opt_crm_prev = copy(dataset_to_train_local_crm)
            dataset_to_opt_crm_prev.idx = range(
                len(dataset_to_train_local_crm.idx) + len(dataset_to_validate_local_crm.idx))
            dataset_to_opt_crm_prev.target = np.append(dataset_to_train_local_crm.target,
                                                       dataset_to_validate_local_crm.target)
            dataset_to_opt_crm_prev.features = np.append(dataset_to_train_local_crm.features,
                                                         dataset_to_validate_local_crm.features, axis=0)

            chain_crm_opt = composer.compose_chain(data=dataset_to_opt_crm_prev,
                                                   initial_chain=None,
                                                   composer_requirements=composer_requirements,
                                                   metrics=metric_function,
                                                   is_visualise=False)
        else:
            from copy import deepcopy
            chain_crm_opt = deepcopy(chain_template)

        # chain_crm_opt = deepcopy(chain_template)
        # chain_crm_opt.fine_tune_primary_nodes(dataset_to_train_local_crm)
        chain_crm_opt.fit(dataset_to_train_local_crm)

        # save_fedot_model(chain_crm_opt, 'chain_crm_opt')

        prediction_crm_opt = chain_crm_opt.predict(dataset_to_validate_local_crm)

        if len(prediction_crm_opt.predict.shape) > 1:
            prediction_crm_opt.predict = prediction_crm_opt.predict[:, -1]

        plt.clf()
        _, ax = plt.subplots()
        plt.plot(prediction_crm_opt.predict, label='ml opt')
        plt.plot(prediction_crm.predict[:, -1], label='ml')
        plt.plot(dataset_to_validate_local_crm.target[199:], label='real')
        mse1 = mse(dataset_to_validate_local_crm.target[199:], prediction_crm.predict[:, -1],
                   squared=False)
        mse2 = mse(dataset_to_validate_local_crm.target[199:], prediction_crm_opt.predict,
                   squared=False)

        plt.title(f'ml {mse1}, evo {mse2}')
        ax.legend()

        plt.savefig(f'D:/tmp/found_{forecasting_step}.png')

        import gc
        gc.collect()

        if not prediction_full:
            prediction_full = prediction
        else:
            prediction_full.idx = np.append(prediction_full.idx, prediction.idx)
            prediction_full.predict = np.append(prediction_full.predict, prediction.predict, axis=0)

        if not prediction_full_crm:
            prediction_full_crm = prediction_crm
        else:
            prediction_full_crm.idx = np.append(prediction_full_crm.idx, prediction_crm.idx)
            prediction_full_crm.predict = np.append(prediction_full_crm.predict, prediction_crm.predict, axis=0)

        if not prediction_full_crm_opt:
            prediction_full_crm_opt = prediction_crm_opt
        else:
            prediction_full_crm_opt.idx = np.append(prediction_full_crm_opt.idx, prediction_crm_opt.idx)
            prediction_full_crm_opt.predict = np.append(prediction_full_crm_opt.predict, prediction_crm_opt.predict)

    rmse_on_valid_simple = calculate_validation_metric(
        prediction_full, prediction_full_crm, prediction_full_crm_opt, dataset_to_validate,
        well_id,
        is_visualise)

    print(well_id)
    print(f'RMSE CRM: {round(rmse_on_valid_simple[0])}')
    print(f'RMSE ML: {round(rmse_on_valid_simple[1])}')
    print(f'RMSE ML with CRM: {round(rmse_on_valid_simple[2])}')
    print(f'Evo RMSE ML with CRM: {round(rmse_on_valid_simple[3])}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from Volve dataset of oil field

    for well in ['5351', '5599', '7078', '7289', '7405f']:
        full_path_train_crm = f'cases/data/oil/oil_crm_X{well}.csv'
        full_path_train_crm = os.path.join(str(project_root()), full_path_train_crm)

        file_path_train = f'cases/data/oil/oil_X{well}.csv'
        full_path_train = os.path.join(str(project_root()), file_path_train)

        run_oil_forecasting_problem(full_path_train,
                                    full_path_train_crm,
                                    forecast_length=100,
                                    max_window_size=100,
                                    is_visualise=True,
                                    well_id=well)
