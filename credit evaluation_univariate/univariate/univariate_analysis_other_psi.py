import pandas as pd
import numpy as np
import math
from scipy import stats
import multiprocessing as mp


def get_table(df):
    """
    :explain: 항목별 PSI 계산을 위한 항목값, 건수, 구성비를 반환합니다.

    :param Series df: 독립변수 Series
    :return dataframe: 항목 건수, 구성비 결과
    """
    table_df = pd.DataFrame(df.value_counts())
    table_df = table_df.reset_index(drop=False)
    table_df.columns = ['value', 'N']
    table_df = table_df.sort_values(by=['value'])
    table_df = table_df.reset_index(drop=True)
    table_df['N'] = table_df['N'].replace(0, 1)
    table_df['ratio'] = table_df['N'] / table_df['N'].sum()
    table_df['value'] = table_df['value'].astype('object')
    return table_df


def calc_psi(binned_train_df, binned_valid_df, binned_test_df,
             binned_model_valid_df, performance_table, base_psi_value, use_columns_list, category_columns_list,
             other_columns_list, other_psi_value):
    """
    :explain: 항복별 PSI 값과 PSI 선정여부 결과를 반환합니다.

    :param dataframe binned_train_df: binning 된 train 데이터
    :param dataframe binned_valid_df: binning 된 valid 데이터
    :param dataframe binned_test_df: binning 된 test 데이터
    :param dataframe binned_model_valid_df: binning 된 model valid 데이터
    :param dataframe performance_table: performance table
    :param float psi_value: PSI 기준값
    :param list use_columns_list: 사용 변수 리스트
    :param list category_columns_list: categorical 변수 리스트
    :param list other_columns_list: PSI 기준값을 다르게 적용할 변수 리스트
    :param float other_psi_value: 특정 변수에 적용할 PSI 기준값
    :return dataframe: 모든 항목의 PSI 값 및 선정여부 결과가 추가된 performance table
    """
    for variable in use_columns_list:

        if variable not in category_columns_list:
            binned_train_df[variable] = binned_train_df[variable].astype('category')
            binned_valid_df[variable] = binned_valid_df[variable].astype('category')
            binned_test_df[variable] = binned_test_df[variable].astype('category')
            binned_model_valid_df[variable] = binned_model_valid_df[variable].astype('category')

        if variable in other_columns_list:
            psi_value = other_psi_value
        else:
            psi_value = base_psi_value

        this_tr_table = get_table(binned_train_df[variable])
        this_tr_table.columns = ['value', 'tr_N', 'tr_ratio']
        this_val_table = get_table(binned_valid_df[variable])
        this_val_table.columns = ['value', 'val_N', 'val_ratio']
        this_tst_table = get_table(binned_test_df[variable])
        this_tst_table.columns = ['value', 'tst_N', 'tst_ratio']
        this_model_valid_table = get_table(binned_model_valid_df[variable])
        this_model_valid_table.columns = ['value', 'val_2019_N', 'val_2019_ratio']

        this_psi_df = pd.merge(this_tr_table, this_val_table, how='outer', on='value')
        this_psi_df = pd.merge(this_psi_df, this_tst_table, how='outer', on='value')
        this_psi_df = pd.merge(this_psi_df, this_model_valid_table, how='outer', on='value')

        this_psi_df['val_psi'] = ((this_psi_df['tr_ratio'] - this_psi_df['val_ratio']) * np.log(
            this_psi_df['tr_ratio'] / this_psi_df['val_ratio']))
        this_psi_df['val_psi'] = this_psi_df['val_psi'].replace(np.inf, 100.0)
        performance_table.loc[variable, 'val_psi'] = this_psi_df['val_psi'].sum()
        performance_table.loc[variable, 'val_psi_안정성'] = (
            'N' if (performance_table.loc[variable, 'val_psi'] >= psi_value) else 'Y')

        this_psi_df['tst_psi'] = ((this_psi_df['tr_ratio'] - this_psi_df['tst_ratio']) * np.log(
            this_psi_df['tr_ratio'] / this_psi_df['tst_ratio']))
        this_psi_df['tst_psi'] = this_psi_df['tst_psi'].replace(np.inf, 100.0)
        performance_table.loc[variable, 'tst_psi'] = this_psi_df['tst_psi'].sum()
        performance_table.loc[variable, 'tst_psi_안정성'] = (
            'N' if (performance_table.loc[variable, 'tst_psi'] >= psi_value) else 'Y')

        this_psi_df['val_2019_psi'] = ((this_psi_df['tr_ratio'] - this_psi_df['val_2019_ratio']) * np.log(
            this_psi_df['tr_ratio'] / this_psi_df['val_2019_ratio']))
        this_psi_df['val_2019_psi'] = this_psi_df['val_2019_psi'].replace(np.inf, 100.0)
        performance_table.loc[variable, 'val_2019_psi'] = this_psi_df['val_2019_psi'].sum()
        performance_table.loc[variable, 'val_2019_psi_안정성'] = (
            'N' if (performance_table.loc[variable, 'val_2019_psi'] >= psi_value) else 'Y')

        performance_table.loc[variable, 'PSI_선정여부'] = (
            'N' if ((performance_table.loc[variable, 'val_psi'] >= psi_value) |
                    (performance_table.loc[variable, 'tst_psi'] >= psi_value) |
                    (performance_table.loc[
                         variable, 'val_2019_psi'] >= psi_value)) else 'Y')

    return performance_table


def calc_top(performance_table, performance_item_list, group_divide_col_name, top_value):
    """
    :explain: PSI 통과한 항목에 한해 정보그룹별 상위 N% 항목 선정 결과를 반환합니다.

    :param dataframe performance_table: performance table
    :param list performance_item_list: iv, ks, ar 리스트 (주로 iv만 사용)
    :param string group_divide_col_name: 정보그룹분류 항목명
    :param float top_value: 변별력 기준값
    :return dataframe: 변별력 선정 결과가 추가된 performance table
    """
    psi_pass_df = performance_table[performance_table['PSI_선정여부'] == 'Y']
    group_list = psi_pass_df[group_divide_col_name].unique().tolist()
    # initialize
    for item in performance_item_list:
        performance_table[item + '_top_YN'] = 'N'

    # select each group top variables
    for group in group_list:
        this_df = psi_pass_df[psi_pass_df[group_divide_col_name] == group]

        threshold_n = int(this_df.shape[0] * top_value)

        for item in performance_item_list:
            perf_df = this_df[[item + '_trn', item + '_val']]
            perf_df = perf_df.sort_values(item + '_trn', ascending=False)

            # 변별력 top_value
            item_top_variables = perf_df.index[:threshold_n].tolist()
            performance_table[item + '_top_YN'][performance_table.index.isin(item_top_variables)] = 'Y'

    top_perf_var_list = []
    for item in performance_item_list:
        top_var = performance_table[performance_table[item + '_top_YN'] == 'Y'].index.tolist()
        top_perf_var_list = top_var + top_perf_var_list

    performance_table['변별력'] = 'N'
    performance_table['변별력'][performance_table.index.isin(top_perf_var_list)] = 'Y'

    return performance_table


def calc_stable(performance_table, performance_item_list, group_divide_col_name, stable_value):
    """
    :explain: 변별력 통과한 항목에 한해 정보그룹별 안정성 ±N% 항목선정 결과를 반환합니다.
              (참고 산술식) train * (1-stable_value) ≤ valid ≤ train * (1 + stable_value)
    :param dataframe performance_table: performance table
    :param list performance_item_list: iv, ks, ar 리스트 (주로 iv만 사용)
    :param string group_divide_col_name: 정보그룹분류 항목명
    :param float stable_value: 안정성 기준값
    :return dataframe: 안정성 선정 결과가 추가된 performance table
    """
    top_pass_df = performance_table[performance_table['변별력'] == 'Y']
    group_list = top_pass_df[group_divide_col_name].unique().tolist()
    # initialize
    for item in performance_item_list:
        performance_table[item + '_stable_YN'] = 'N'

    stable_var_list = []
    for group in group_list:
        this_df = top_pass_df[top_pass_df[group_divide_col_name] == group]

        for item in performance_item_list:
            stable_df = this_df[[item + '_trn', item + '_val', item + '_tst']].copy()
            stable_df[item + '_lower'] = (1 - stable_value) * stable_df[item + '_trn']
            stable_df[item + '_upper'] = (1 + stable_value) * stable_df[item + '_trn']

            lower_bool_val = np.array(stable_df[item + '_lower'] <= stable_df[item + '_val'])
            upper_bool_val = np.array(stable_df[item + '_val'] <= stable_df[item + '_upper'])

            #         lower_bool_tst = np.array(stable_df[item + '_lower'] <= stable_df[item + '_tst'])
            #         upper_bool_tst = np.array(stable_df[item + '_tst'] <= stable_df[item + '_upper'])

            satisfied_val = lower_bool_val & upper_bool_val

            #         satisfied_tst = lower_bool_tst & upper_bool_tst
            #         satisfied = satisfied_val & satisfied_tst

            satisfied = satisfied_val
            stable_var = stable_df.index[satisfied].tolist()
            performance_table[item + '_stable_YN'][performance_table.index.isin(stable_var)] = 'Y'

            # update
            stable_var_list = stable_var + stable_var_list

    final_stable_var_list = []
    for item in performance_item_list:
        stable_var = performance_table[performance_table[item + '_stable_YN'] == 'Y'].index.tolist()
        final_stable_var_list = stable_var + final_stable_var_list

    performance_table['안정성'] = 'N'
    performance_table['안정성'][performance_table.index.isin(final_stable_var_list)] = 'Y'

    return performance_table


def object_to_code(binned_df):
    """
    :explain: bin값을 정수(코드화)로 변환하는 함수입니다.

    :param dataframe binned_df: binning된 데이터
    :return dataframe: bin값이 정수로 변환된 결과
    """
    category = pd.Categorical(binned_df)
    result = category.codes
    return result


def calc_correlation(performance_table, group_divide_colname, corr_value, binned_train_df):
    """
    :explain: 안정성 통과한 항목에 한해 정보그룹별 상관분석 항목선정 결과를 반환합니다.

    :param dataframe performance_table: performance table
    :param string group_divide_colname: 정보그룹분류 항목명
    :param float corr_value: 상관분석 기준값
    :param dataframe binned_train_df: binning 된 train 데이터
    :return dataframe: 상관분석 결과가 추가된 performance table
    """
    stable_pass_df = performance_table[performance_table['안정성'] == 'Y']
    group_list = stable_pass_df[group_divide_colname].unique().tolist()

    high_col_list = []
    i = 0
    basic_corr_dict = dict()
    result_rmv_corr_dict = dict()
    for group in group_list:
        i += 1
        this_df = stable_pass_df[stable_pass_df[group_divide_colname] == group]

        # spearman correlation을 기준으로 변수 선택
        # train의 iv가 큰 애들을 살림
        this_df = this_df.sort_values('iv_trn', ascending=False)

        cols = this_df.index

        if ((len(cols) == 0) | (len(cols) == 1)) & (i == len(group_list)):
            break
        elif (len(cols) == 0) | (len(cols) == 1):
            continue

        cor_trn = binned_train_df[cols].copy()

        # convert code
        cor_trn = cor_trn.apply(object_to_code)

        # 대각 행렬 + lower triangluar : 0

        correlation_df = np.asarray(stats.spearmanr(cor_trn)[0])
        if correlation_df.ndim == 0:
            correlation_df = pd.DataFrame([[0, stats.spearmanr(cor_trn)[0]], [stats.spearmanr(cor_trn)[0], 0]])
            correlation_df.index = cols
            correlation_df.columns = cols
        else:
            correlation_df = abs(np.triu(stats.spearmanr(cor_trn)[0]))
            np.fill_diagonal(correlation_df, 0)

            correlation_df = pd.DataFrame(correlation_df)
            correlation_df.index = cols
            correlation_df.columns = cols

        basic_corr_dict[str(group)] = correlation_df

        # correlation > cor_p가 될 때까지 여러번 시행
        while np.sum(np.sum(correlation_df > corr_value)) > 0:
            if correlation_df.shape[0] == 2:
                high_corr_var_idx, high_corr_var = [], []
                high_corr_var_one = correlation_df.columns[1]
                high_corr_var.append(high_corr_var_one)
                # cor 높은 col 삭제
                correlation_df = correlation_df.drop(high_corr_var, axis=0)
                correlation_df = correlation_df.drop(high_corr_var, axis=1)
                # print(len(high_corr_var))\

                # update
                high_col_list = high_corr_var + high_col_list

            else:
                # corr > cor_p이면 삭제
                high_corr_var_idx = np.where(correlation_df > corr_value)
                high_corr_var_idx = pd.DataFrame({'A': high_corr_var_idx[0], 'B': high_corr_var_idx[1]})
                high_corr_var_idx = high_corr_var_idx[~(high_corr_var_idx['A'].isin(high_corr_var_idx['B']))]
                high_corr_var_idx = high_corr_var_idx['B'].unique()
                high_corr_var = correlation_df.columns[high_corr_var_idx].tolist()

                # cor 높은 col 삭제
                correlation_df = correlation_df.drop(high_corr_var, axis=0)
                correlation_df = correlation_df.drop(high_corr_var, axis=1)

                # update
                high_col_list = high_corr_var + high_col_list

        result_rmv_corr_dict[str(group)] = correlation_df

    performance_table['상관분석'] = 'Y'
    performance_table['상관분석'][(performance_table.index.isin(high_col_list)) | (performance_table['안정성'] == 'N')] = 'N'

    return performance_table, basic_corr_dict, result_rmv_corr_dict


def select_candidate_variable(binned_train_df, binned_valid_df, binned_test_df,
                              binned_model_valid_df, performance_table, use_columns_list, category_columns_list,
                              other_columns_list, other_psi_value,
                              performance_item_list, group_divide_col_name,
                              base_psi_value, top_value, stable_value, corr_value):
    """
    :explain: 단변량 분석을 통한 후보변수 선정 결과 및 상관분석 결과를 반환합니다.

    :param dataframe binned_train_df: binning 된 train 데이터
    :param dataframe binned_valid_df: binning 된 valid 데이터
    :param dataframe binned_test_df: binning 된 test 데이터
    :param dataframe binned_model_valid_df: binning 된 model valid 데이터
    :param dataframe performance_table: performance table
    :param list use_columns_list: 사용 변수 리스트
    :param list category_columns_list: categorical 변수 리스트
    :param list other_columns_list: PSI 기준값을 다르게 적용할 변수 리스트
    :param float other_psi_value: 특정 변수에 적용할 PSI 기준값
    :param list performance_item_list: iv, ks, ar 리스트 (주로 iv만 사용)
    :param string group_divide_col_name: 정보그룹분류 항목명
    :param float base_psi_value: PSI 기준값
    :param float top_value: 변별력 기준값
    :param float stable_value: 안정성 기준값
    :param float corr_value: 상관분석 기준값
    :return tuple: 후보변수 선정 dataframe, 상관분석 dictionary, 상관성 높은 변수 제거한 후 상관분석 dictionary
    """
    psi_result = calc_psi(binned_train_df, binned_valid_df, binned_test_df,
                          binned_model_valid_df, performance_table, base_psi_value, use_columns_list, category_columns_list,
                          other_columns_list, other_psi_value)
    top_result = calc_top(psi_result, performance_item_list, group_divide_col_name, top_value)
    stable_result = calc_stable(top_result, performance_item_list, group_divide_col_name, stable_value)
    corr_result = calc_correlation(stable_result, group_divide_col_name, corr_value, binned_train_df)

    final_performance_table = corr_result[0]
    basic_corr_dict = corr_result[1]
    final_rmv_corr_dict = corr_result[2]
    return final_performance_table, basic_corr_dict, final_rmv_corr_dict