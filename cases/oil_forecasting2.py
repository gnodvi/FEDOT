import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root


def get_ms_chain():
    chain = Chain()
    node_trend = PrimaryNode('trend_data_model')
    node_lstm_trend = SecondaryNode('rfr', nodes_from=[node_trend])

    node_residual = PrimaryNode('residual_data_model')
    node_ridge_residual = SecondaryNode('rfr', nodes_from=[node_residual])

    node_final = SecondaryNode('additive_data_model',
                               nodes_from=[node_ridge_residual, node_lstm_trend])
    chain.add_node(node_final)
    return chain


def get_comp_chain():
    chain = Chain()
    node_1 = PrimaryNode('rfr')
    # node_lstm_trend = SecondaryNode('rfr', nodes_from=[node_trend])

    node_2 = PrimaryNode('dtreg')
    # node_ridge_residual = SecondaryNode('rfr', nodes_from=[node_residual])

    node_final = SecondaryNode('linear',
                               nodes_from=[node_1, node_2])
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, pred_crm, pred_crm_opt, valid: InputData,
                                name: str, is_visualise=False):
    forecast_length = pred.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[:, pred.predict.shape[1] - 1]
    predicted_crm = pred_crm.predict[:, pred_crm.predict.shape[1] - 1]
    predicted_crm_opt = pred_crm_opt.predict  # [:, pred_crm_opt.predict.shape[1] - 1]

    real = valid.target[max(len(valid.target) - len(predicted), 0):]

    file_path_crm = f'D:/THEODOR/cases/data/oil/crimp.csv'

    data_frame = pd.read_csv(file_path_crm, sep=',')
    crm = data_frame[f'mean_{name}'][(300 - 4):700]
    crm[np.isnan(crm)] = 0
    # the quality assessment for the simulation results
    rmse_ml = mse(y_true=real, y_pred=predicted, squared=False)
    rmse_ml_crm = mse(y_true=real, y_pred=predicted_crm, squared=False)
    rmse_crm_opt = mse(y_true=real, y_pred=predicted_crm_opt, squared=False)
    rmse_crm = mse(y_true=real, y_pred=crm, squared=False)

    from dtw import dtw
    dist = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    dtw_ml, _, _, _ = dtw(real, predicted, dist)
    dtw_ml_crm, _, _, _ = dtw(real, predicted_crm, dist)
    dtw_crm_opt, _, _, _ = dtw(real, predicted_crm_opt, dist)
    dtw_crm, _, _, _ = dtw(real, crm, dist)

    # plot results
    if is_visualise:
        compare_plot(predicted, predicted_crm, predicted_crm_opt, real,
                     forecast_length=forecast_length,
                     model_name=name, err=rmse_crm)
    return rmse_crm, rmse_ml, rmse_ml_crm, rmse_crm_opt, \
           dtw_crm, dtw_ml, dtw_ml_crm, dtw_crm_opt


def compare_plot(predicted, predicted_crm, predicted_crm_opt, real, forecast_length, model_name, err):
    file_path_crm = f'D:/THEODOR/cases/data/oil/crimp.csv'

    data_frame = pd.read_csv(file_path_crm, sep=',')
    mean = data_frame[f'mean_{model_name}'][300:700]
    min_int = data_frame[f'min_{model_name}'][300:700]
    max_int = data_frame[f'max_{model_name}'][300:700]

    times = [_ for _ in range(len(mean))]

    plt.clf()
    _, ax = plt.subplots()
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

    dataset_to_train_local_crm_full = None
    dataset_to_train_local_full = None

    for forecasting_step in range(4):
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

        node_1 = PrimaryNode('rfr')
        chain_simple = Chain(node_1)

        node_single2 = PrimaryNode('rfr')
        chain_simple_crm = Chain(node_single2)

        # node_1 = PrimaryNode('rfr')
        # node_2 = PrimaryNode('dtreg')
        # node_single = SecondaryNode('knnreg', nodes_from=[node_1, node_2])
        chain_crm_opt = get_comp_chain()  # Chain(node_single)

        if not dataset_to_train_local_full:
            dataset_to_train_local_full = dataset_to_train_local
        else:
            dataset_to_train_local_full.target = np.append(dataset_to_train_local_full.target,
                                                           dataset_to_train_local.target)
            dataset_to_train_local_full.features = np.append(dataset_to_train_local_full.features,
                                                             dataset_to_train_local.features, axis=0)
            dataset_to_train_local_full.idx = np.append(dataset_to_train_local_full.idx,
                                                        dataset_to_train_local.idx)
        if not dataset_to_train_local_crm_full:
            dataset_to_train_local_crm_full = dataset_to_train_local_crm
        else:
            dataset_to_train_local_crm_full.features = np.append(dataset_to_train_local_crm_full.features,
                                                                 dataset_to_train_local_crm.features, axis=0)
            dataset_to_train_local_crm_full.target = np.append(dataset_to_train_local_crm_full.target,
                                                               dataset_to_train_local_crm.target)
            dataset_to_train_local_crm_full.idx = np.append(dataset_to_train_local_crm_full.idx,
                                                            dataset_to_train_local_crm.idx)

        chain_simple.fit(input_data=dataset_to_train_local_full, verbose=False)
        chain_simple_crm.fit(input_data=dataset_to_train_local_crm_full, verbose=False)
        chain_crm_opt.fit(input_data=dataset_to_train_local_crm_full, verbose=False)

        prediction = chain_simple.predict(dataset_to_validate_local)
        prediction_crm = chain_simple_crm.predict(dataset_to_validate_local_crm)
        prediction_crm_opt = chain_crm_opt.predict(dataset_to_validate_local_crm)

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
            prediction_full_crm_opt.predict = np.append(prediction_full_crm_opt.predict, prediction_crm_opt.predict,
                                                        axis=0)

    rmse_on_valid_simple = calculate_validation_metric(
        prediction_full, prediction_full_crm, prediction_full_crm_opt, dataset_to_validate,
        well_id,
        is_visualise)

    print(well_id)
    print(f'RMSE CRM: {round(rmse_on_valid_simple[0])}')
    print(f'RMSE ML: {round(rmse_on_valid_simple[1])}')
    print(f'RMSE ML with CRM: {round(rmse_on_valid_simple[2])}')
    print(f'Evo RMSE ML with CRM: {round(rmse_on_valid_simple[3])}')

    print(f'DTW CRM: {round(rmse_on_valid_simple[4])}')
    print(f'DTW ML: {round(rmse_on_valid_simple[5])}')
    print(f'DTW ML with CRM: {round(rmse_on_valid_simple[6])}')
    print(f'DTW RMSE ML with CRM: {round(rmse_on_valid_simple[7])}')

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
