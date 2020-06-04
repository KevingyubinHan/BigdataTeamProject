import pandas as pd
import numpy as np
import multiprocessing as mp
from univariate.calc_variable_statistics import calc_variable_stats_func


def merge_bin(x_binned_df, variable, y_df, y_column_name):
    """
    :explain: 해당 변수의 모든 bin에 대해서 IV 변화율이 가장 낮으면서  bin이 10개가 되도록 변환하는 함수입니다.

    :param dataframe x_binned_df: fine classing을 완료한 독립변수 데이터
    :param string variable: 독립변수명
    :param dataframe y_df: 종속변수 데이터
    :param string y_column_name: 종속변수명
    :return dataframe: bin을 최적의 10개까지 결합한 통계량 결과
    """
    original_iv_result = calc_variable_stats_func(x_binned_df, variable, y_df, y_column_name)
    original_iv_df = original_iv_result['df']

    iv_df = original_iv_df.copy()

    s = 0
    while iv_df.shape[0] > 10:
        s += 1

        if s == 1:
            sum_iv_list = []
            iv_list = []
            for idx in range(iv_df.shape[0] - 1):
                remain_idx_df = iv_df.drop([idx, idx + 1])
                remain_idx_df = remain_idx_df[['Distribution Good', 'Distribution Bad', 'WoE', 'IV']]
                grouping_idx_df = iv_df.loc[[idx, idx + 1]]

                new_df_dict = {
                    'Distribution Good': np.sum(grouping_idx_df['Good']) / np.sum(iv_df['Good']),
                    'Distribution Bad': np.sum(grouping_idx_df['Bad']) / np.sum(iv_df['Bad']),
                }

                new_df = pd.DataFrame(new_df_dict, index=[idx])
                new_df['WoE'] = np.log(new_df['Distribution Good'] / new_df['Distribution Bad'])
                new_df['IV'] = new_df['WoE'] * (new_df['Distribution Good'] - new_df['Distribution Bad'])
                merged_df = pd.concat([remain_idx_df, new_df], axis=0)

                iv_list.append(merged_df['IV'][idx])
                sum_iv_list.append(np.sum(merged_df['IV']))

            best_idx = np.where(sum_iv_list == np.max(sum_iv_list))[0].tolist()[0]
            iv_df = get_iv_df(iv_df, best_idx, variable)
        else:

            if best_idx == 0:
                remove_idx_list = [best_idx, best_idx + 1]
                add_idx_list = [best_idx]
            else:
                if best_idx == (len(iv_list) - 1):
                    remove_idx_list = [best_idx - 1, best_idx]
                    add_idx_list = [best_idx - 1]
                else:
                    remove_idx_list = [best_idx - 1, best_idx, best_idx + 1]
                    add_idx_list = [best_idx - 1, best_idx]

            remove_idx_list.sort(reverse=True)
            for rm_idx in remove_idx_list:
                iv_list.pop(rm_idx)

            for idx in sorted(add_idx_list):
                remain_idx_df = iv_df.drop([idx, idx + 1])
                remain_idx_df = remain_idx_df[['Distribution Good', 'Distribution Bad', 'WoE', 'IV']]
                grouping_idx_df = iv_df.loc[[idx, idx + 1]]

                new_df_dict = {
                    'Distribution Good': np.sum(grouping_idx_df['Good']) / np.sum(iv_df['Good']),
                    'Distribution Bad': np.sum(grouping_idx_df['Bad']) / np.sum(iv_df['Bad'])
                }

                new_df = pd.DataFrame(new_df_dict, index=[idx])
                new_df['WoE'] = np.log(new_df['Distribution Good'] / new_df['Distribution Bad'])
                new_df = new_df.replace({'WoE': {np.inf: 0, -np.inf: 0}})
                new_df['IV'] = new_df['WoE'] * (new_df['Distribution Good'] - new_df['Distribution Bad'])

                merged_df = pd.concat([remain_idx_df, new_df], axis=0)

                # update
                iv_list.insert(idx, merged_df['IV'][idx])

            iv_df_list = iv_df['IV']
            sum_iv_list = []
            for i in range(len(iv_df_list) - 1):
                sum_iv_list.append(iv_df_list.drop([i, i + 1]).sum() + iv_list[i])

            best_idx = np.where(sum_iv_list == np.max(sum_iv_list))[0].tolist()[0]

            iv_df = get_iv_df(iv_df, best_idx, variable)

    return iv_df


def get_iv_df(iv_df, best_idx, variable):
    """
    :explain: 두 개의 bin을 결합했을 때의 statistics 결과를 반환합니다.

    :param dataframe iv_df: bin 결합 전 statistics 결과
    :param array best_idx: 결합할 bin의 statistics기준 index
    :param string variable: 독립변수명
    :return dataframe: bin 결합 후 statistics 결과
    """
    remain_idx_df = iv_df.drop([best_idx, best_idx + 1])
    grouping_idx_df = iv_df.loc[[best_idx, best_idx + 1]]

    new_df_dict = {
        'Variable': variable,
        'Value': pd.Interval(left=grouping_idx_df.loc[best_idx]['Value'].left,
                             right=grouping_idx_df.loc[best_idx + 1]['Value'].right),
        'All': np.sum(grouping_idx_df['All']),
        'Good': np.sum(grouping_idx_df['Good']),
        'Bad': np.sum(grouping_idx_df['Bad']),
        'Cum_Good': np.max(grouping_idx_df['Cum_Good']),
        'Cum_Bad': np.max(grouping_idx_df['Cum_Bad']),
        'Cum_Total': np.max(grouping_idx_df['Cum_Total']),
        'Total_ratio': np.sum(grouping_idx_df['All']) / np.sum(iv_df['All']),
        'Bad Rate': np.sum(grouping_idx_df['Bad']) / np.sum(grouping_idx_df['All']),
        'Distribution Good': np.sum(grouping_idx_df['Good']) / np.sum(iv_df['Good']),
        'Distribution Bad': np.sum(grouping_idx_df['Bad']) / np.sum(iv_df['Bad']),
        'Cum_Good_p': np.max(grouping_idx_df['Cum_Good']) / np.sum(iv_df['Good']),
        'Cum_Bad_p': np.max(grouping_idx_df['Cum_Bad']) / np.sum(iv_df['Bad']),
        'Cum_Total_p': np.max(grouping_idx_df['Cum_Total']) / np.sum(iv_df['All'])
    }

    new_df = pd.DataFrame(new_df_dict, index=[best_idx])
    new_df['WoE'] = np.log(new_df['Distribution Good'] / new_df['Distribution Bad'])
    new_df = new_df.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    new_df['IV'] = new_df['WoE'] * (new_df['Distribution Good'] - new_df['Distribution Bad'])
    new_df['KS'] = abs(new_df['Cum_Good_p'] - new_df['Cum_Bad_p']) * 100

    new_iv_df = pd.concat([remain_idx_df, new_df], axis=0)
    new_iv_df = new_iv_df.sort_index()
    new_iv_df = new_iv_df.reset_index(drop=True)

    return new_iv_df


def grouping(fine_binned_df, coarse_statistics_result_df, variable):
    """
    :explain: bin 결합 후 데이터에 결합한 bin 결과를 적용하는 함수입니다.

    :param dataframe fine_binned_df: fine classing 완료한 독립변수 데이터
    :param dataframe coarse_statistics_result_df: bin 결합한 statistics 결과 데이터
    :param string variable: 독립변수명
    :return dataframe: coarse classing 결과를 적용한 binning dataframe
    """

    def right_return(df):
        result = df.right
        return result

    def interval_in_check(df, idx):
        result = df in coarse_statistics_result_df['Value'][idx]
        return result

    right_value_df = fine_binned_df[variable].apply(right_return)
    for idx in range(coarse_statistics_result_df.shape[0]):
        fine_binned_df[variable] = np.where(right_value_df.apply(interval_in_check, idx=idx),
                                            coarse_statistics_result_df['Value'][idx],
                                            fine_binned_df[variable])
    return fine_binned_df


def coarse_classing(x_binned_train_df,
                    y_train_df,
                    x_binned_valid_df,
                    x_binned_test_df,
                    x_binned_model_valid_df,
                    y_column_name):
    """
    :explain: 각각의 bin을 결합해보며 IV 변화율이 가장 낮고 bin개수가 10개일 때의 bin 결합조합을 적용한 결과를 반환합니다.

    :param dataframe x_binned_train_df: fine classing 완료한 train 독립변수 데이터
    :param dataframe y_train_df: train 종속변수 데이터
    :param dataframe x_binned_valid_df: fine classing 완료한 valid 독립변수 데이터
    :param dataframe x_binned_test_df: fine classing 완료한 test 독립변수 데이터
    :param dataframe x_binned_model_valid_df: fine classing 완료한 model valid 독립변수 데이터
    :param string y_column_name: 종속변수명
    :return tuple: coarse classing된 train/valid/test/model valid dataframe
    """
    use_cols = x_binned_train_df.columns.tolist()
    bin_cnt = x_binned_train_df.apply(lambda x: len(pd.unique(x)))
    coarse_variable_list = bin_cnt[bin_cnt > 10].keys().tolist()

    for variable in use_cols:
        if variable in coarse_variable_list:
            coarse_classing_df = merge_bin(x_binned_train_df, variable, y_train_df, y_column_name)

            binned_trn_x = grouping(x_binned_train_df, coarse_classing_df, variable)
            binned_val_x = grouping(x_binned_valid_df, coarse_classing_df, variable)
            binned_tst_x = grouping(x_binned_test_df, coarse_classing_df, variable)
            binned_model_valid_x = grouping(x_binned_model_valid_df, coarse_classing_df, variable)

    return binned_trn_x, binned_val_x, binned_tst_x, binned_model_valid_x


def parallelize_coarse_classing(x_binned_train_df, y_train_df,
                                    x_binned_valid_df,
                                    x_binned_test_df,
                                    x_binned_model_valid_df,
                                    y_column_name, func, processes):
    """
    :explain: 병렬처리를 적용하여 coarse classing 결과를 반환합니다.

    :param dataframe x_binned_train_df: fine classing 완료한 train 독립변수 데이터
    :param dataframe y_train_df: train 종속변수 데이터
    :param dataframe x_binned_valid_df: fine classing 완료한 valid 독립변수 데이터
    :param dataframe x_binned_test_df: fine classing 완료한 test 독립변수 데이터
    :param dataframe x_binned_model_valid_df: fine classing 완료한 model valid 독립변수 데이터
    :param string y_column_name: 종속변수명
    :param function func: 병렬처리할 coarse_classing 함수
    :param int processes: 병렬처리에 사용할 core 개수
    :return tuple: coarse classing된 train/valid/test/model valid dataframe
    """
    use_cols = x_binned_train_df.columns.tolist()
    split_cols = np.array_split(use_cols, processes, axis=0)
    arg_tuple_list = [(x_binned_train_df[split_col], y_train_df, x_binned_valid_df[split_col],
                       x_binned_test_df[split_col], x_binned_model_valid_df[split_col], y_column_name)\
                      for split_col in split_cols]

    pool = mp.Pool(processes=processes)
    result = pool.starmap(func, arg_tuple_list)
    pool.close()
    pool.join()

    binned_trn_x_list = []
    binned_val_x_list = []
    binned_tst_x_list = []
    binned_model_val_x_list = []

    for value in result:
        binned_trn_x_list.append(value[0])
        binned_val_x_list.append(value[1])
        binned_tst_x_list.append(value[2])
        binned_model_val_x_list.append(value[3])

    binned_trn_x = pd.concat(binned_trn_x_list, axis=1)
    binned_val_x = pd.concat(binned_val_x_list, axis=1)
    binned_tst_x = pd.concat(binned_tst_x_list, axis=1)
    binned_model_val_x = pd.concat(binned_model_val_x_list, axis=1)

    return binned_trn_x, binned_val_x, binned_tst_x, binned_model_val_x
