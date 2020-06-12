from copy import copy
from datetime import timedelta

import numpy as np

from core.models.data import InputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import ModelMetaInfo, ModelTypesRepository
from core.repository.tasks import Task, TaskTypesEnum, compatible_task_types

DEFAULT_PARAMS_STUB = 'default_params'


class Model:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self._eval_strategy, self._data_preprocessing = None, None
        self.params = DEFAULT_PARAMS_STUB

    @property
    def acceptable_task_types(self):
        model_info = ModelTypesRepository().model_info_by_id(self.model_type)
        return model_info.task_type

    def compatible_task_type(self, base_task_type: TaskTypesEnum):
        # if the model can't be used directly for the task type from data
        if base_task_type not in self.acceptable_task_types:
            # search the supplementary task types, that can be included in chain which solves original task
            globally_compatible_task_types = compatible_task_types(base_task_type)
            compatible_task_types_acceptable_for_model = list(set(self.acceptable_task_types).intersection
                                                              (set(globally_compatible_task_types)))
            if len(compatible_task_types_acceptable_for_model) == 0:
                raise ValueError(f'Model {self.model_type} can not be used as a part of {base_task_type}.')
            task_type_for_model = compatible_task_types_acceptable_for_model[0]
            return task_type_for_model
        return base_task_type

    @property
    def metadata(self) -> ModelMetaInfo:
        model_info = ModelTypesRepository().model_info_by_id(self.model_type)
        if not model_info:
            raise ValueError(f'Model {self.model_type} not found')
        return model_info

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        else:
            return output_types[0]

    @property
    def description(self):
        model_type = self.model_type
        model_params = self.params
        return f'n_{model_type}_{model_params}'

    def _init(self, task: Task):

        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        self._eval_strategy = _eval_strategy_for_task(self.model_type, task.task_type)(self.model_type, params_for_fit)

    def fit(self, data: InputData):
        self._init(data.task)

        data_for_fit = _drop_data_with_nan(data)

        fitted_model = self._eval_strategy.fit(train_data=data_for_fit)

        predict_train = self.predict(fitted_model, data)

        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        self._init(data.task)

        data_for_predict = _drop_data_with_nan(data, ignore_nan_in_target=True)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=data_for_predict)

        prediction = _post_process_prediction(prediction, data.task, len(data.idx))

        return prediction

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        self._init(data.task)

        data_for_fit = _drop_data_with_nan(data, ignore_nan_in_target=True)

        try:
            fitted_model, tuned_params = self._eval_strategy.fit_tuned(train_data=data_for_fit,
                                                                       iterations=iterations,
                                                                       max_lead_time=max_lead_time)
            self.params = tuned_params
            if not self.params:
                self.params = DEFAULT_PARAMS_STUB
        except Exception as ex:
            print(f'Tuning failed because of {ex}')
            fitted_model = self._eval_strategy.fit(train_data=data_for_fit)
            self.params = DEFAULT_PARAMS_STUB

        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=data_for_fit)

        predict_train = _post_process_prediction(predict_train, data.task, len(data.idx))

        return fitted_model, predict_train

    def __str__(self):
        return f'{self.model_type}'


def _eval_strategy_for_task(model_type: str, task_type_for_data: TaskTypesEnum):
    models_repo = ModelTypesRepository()
    model_info = models_repo.model_info_by_id(model_type)

    task_type_for_model = task_type_for_data
    task_types_acceptable_for_model = model_info.task_type

    # if the model can't be used directly for the task type from data
    if task_type_for_model not in task_types_acceptable_for_model:
        # search the supplementary task types, that can be included in chain which solves original task
        globally_compatible_task_types = compatible_task_types(task_type_for_model)
        compatible_task_types_acceptable_for_model = list(set(task_types_acceptable_for_model).intersection
                                                          (set(globally_compatible_task_types)))
        if len(compatible_task_types_acceptable_for_model) == 0:
            raise ValueError(f'Model {model_type} can not be used as a part of {task_type_for_model}.')
        task_type_for_model = compatible_task_types_acceptable_for_model[0]

    strategy = models_repo.model_info_by_id(model_type).current_strategy(task_type_for_model)
    return strategy


def _post_process_prediction(prediction, task: Task, expected_length: int):
    if np.array([np.isnan(_) for _ in prediction]).any():
        prediction = np.nan_to_num(prediction)

    if task.task_type == TaskTypesEnum.ts_forecasting:
        prediction = _post_process_ts_prediction(prediction, task, expected_length)

    return prediction


def _post_process_ts_prediction(prediction, task: Task, expected_length: int):
    if not task.task_params.return_all_steps and len(prediction.shape) > 1:
        # choose last forecasting step only for each prediction
        prediction = prediction[:, -1]

    if not task.task_params.make_future_prediction:
        # cut unwanted oos predection
        length_of_cut = (task.task_params.forecast_length - 1)
        if length_of_cut > 0:
            prediction = prediction[:-length_of_cut]
    # TODO add multivariate

    # add zeros to preserve length
    if len(prediction) < expected_length:
        zeros = np.zeros((expected_length - len(prediction), *prediction.shape[1:]))
        zeros = [np.nan] * len(zeros)

        if len(prediction.shape) == 1:
            prediction = np.concatenate((zeros, prediction))
        else:
            prediction_steps = []
            for forecast_depth in range(prediction.shape[1]):
                prediction_steps.append(np.concatenate((zeros, prediction[:, forecast_depth])))
            prediction = np.stack(np.asarray(prediction_steps)).T
    return prediction


def _drop_data_with_nan(data_to_clean: InputData, ignore_nan_in_target: bool = False):
    data_to_clean = _clean_nans(data_to_clean, data_to_clean.features)

    if not ignore_nan_in_target:
        # can be acceptable in prediction
        data_to_clean = _clean_nans(data_to_clean, data_to_clean.target)

    return data_to_clean


def _clean_nans(data: InputData, array_with_nans: np.ndarray):
    data_to_clean = copy(data)
    if array_with_nans is None:
        return data_to_clean
    # remove all rows with nan in array_with_nans
    if len(array_with_nans.shape) == 1:
        data_to_clean.idx = data.idx[~np.isnan(array_with_nans)]
        if data.features is not None:
            data_to_clean.features = data.features[~np.isnan(array_with_nans)]
        if data.target is not None:
            data_to_clean.target = data.target[~np.isnan(array_with_nans)]
    elif len(array_with_nans.shape) == 2:
        data_to_clean.idx = data.idx[~np.isnan(array_with_nans).any(axis=1)]
        if data.features is not None:
            data_to_clean.features = data.features[~np.isnan(array_with_nans).any(axis=1)]
        if data.target is not None:
            data_to_clean.target = data.target[~np.isnan(array_with_nans).any(axis=1)]
    elif len(array_with_nans.shape) == 3:
        for dim in range(array_with_nans.shape[2]):
            data_to_clean.idx = data.idx[~np.isnan(array_with_nans).any(axis=1).any(axis=1)]
            if data.features is not None:
                data_to_clean.features = \
                    data.features[~np.isnan(array_with_nans[:, :, dim]).any(axis=1)]
            if data.target is not None:
                data_to_clean.target = \
                    data.target[~np.isnan(array_with_nans[:, :, dim]).any(axis=1)]

    return data_to_clean
