import pandas as pd
import numpy as np
import multiprocessing as mp


def calc_variable_stats_func(x_binned_df, variable, y_df, y_column_name):
    """
    :explain: 입력 받은 변수에 대한 statistics 결과를 반환합니다.

    :param dataframe x_binned_df: coarse classing 완료한 독립변수 데이터
    :param string variable: 독립변수명
    :param dataframe y_df: 종속변수 데이터
    :param string y_column_name: 종속변수명
    :return dictionary: 각 항목별 통계량 결과
    """
    df = pd.concat([y_df, x_binned_df], axis=1)

    data = df.groupby(by=[variable], as_index=False).agg(
        {
            y_column_name: ('count', 'sum')
        }
    )
    data.columns = ['Value', 'All', 'Bad']
    data['Variable'] = variable
    data['Good'] = data['All'] - data['Bad']
    data = data[['Variable', 'Value', 'All', 'Good', 'Bad']]

    data = data.sort_values(by=['Value'], ascending=[True])
    data.index = range(len(data.index))

    data['Cum_Good'] = np.cumsum(data['Good'])
    data['Cum_Bad'] = np.cumsum(data['Bad'])
    data['Cum_Total'] = data['Cum_Good'] + data['Cum_Bad']

    data['Total_ratio'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()

    data['Cum_Good_p'] = np.cumsum(data['Good']) / data['Good'].sum()
    data['Cum_Bad_p'] = np.cumsum(data['Bad']) / data['Bad'].sum()
    data['Cum_Total_p'] = np.cumsum(data['Total_ratio'])

    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    data['KS'] = abs(data['Cum_Good_p'] - data['Cum_Bad_p']) * 100

    cum_bad = list(data['Cum_Bad_p'])
    cum_good = list(data['Cum_Good_p'])
    cum_bad.insert(0, 0)
    cum_good.insert(0, 0)
    cum_bad = np.array(cum_bad)
    cum_good = np.array(cum_good)

    x = cum_good[1:] - cum_good[:-1]
    y = cum_bad[1:] + cum_bad[:-1]

    ar = abs(np.sum(x * y) - 1) * 100

    iv = data['IV'].sum()
    ks = np.max(data['KS'])

    result = {'IV': iv,
              'KS': ks,
              'AR': ar,
              'df': data}
    return result


def calc_variable_stats(x_binned_df, use_columns_list, y_df, y_column_name):
    """
    :explain: 입력 받은 모든 사용 변수 대해 statistics 결과를 반환합니다.

    :param dataframe x_binned_df: coarse classing 완료한 독립변수 데이터
    :param list use_columns_list: 사용할 변수 리스트
    :param dataframe y_df: 종속변수 데이터
    :param string y_column_name: 종속변수명
    :return dictionary: 모든 사용변수의 statistics 결과
    """
    calc_variable_stats_dict = dict()
    for col in use_columns_list:
        result = calc_variable_stats_func(x_binned_df, col, y_df, y_column_name)
        calc_variable_stats_dict[col] = result
    return calc_variable_stats_dict


def merge_dicts(dict_list):
    """
    :explain: 병렬처리 진행한 dictionary 결과를 결합하는 함수입니다.

    :param list dict_list: statistics 계산 완료한 dictionary 결과 리스트
    :return dictionary: 병렬처리 outputs을 합친 결과
    """
    result = {}
    for dictionary in dict_list:
        result.update(dictionary)
    return result


def parallelize_calc_variable_stats(x_binned_df, use_columns_list, y_df, y_column_name, func, processes):
    """
    :explain: 병렬처리를 적용하여 statistics 결과를 변환합니다.

    :param dataframe x_binned_df: coarse classing 완료한 독립변수 데이터
    :param list use_columns_list: 사용할 변수 리스트
    :param dataframe y_df: 종속변수 데이터
    :param string y_column_name: 종속변수명
    :param function func: 병렬처리할 calc_variable_stats 함수
    :param int processes: 병렬처리에 사용할 core 개수
    :return dictionary: 각각의 변수에 대한 statistics 결과
    """
    split_cols = np.array_split(use_columns_list, processes, axis=0)
    pool = mp.Pool(processes=processes)
    arg_tuple_list = [(x_binned_df[split_col], split_col, y_df, y_column_name) for split_col in split_cols]
    result = pool.starmap(func, arg_tuple_list)
    pool.close()
    pool.join()

    calc_stats_result_dict = merge_dicts(result)
    return calc_stats_result_dict
