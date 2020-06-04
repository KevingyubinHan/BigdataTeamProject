import pandas as pd
import numpy as np
import multiprocessing as mp


def fine_classing(x_train_df, x_valid_df, x_test_df, x_model_valid_df, numeric_columns_list, bin_cut):
    """
    :explain: binning 범위를 입력 받아 numeric 변수에 한해 train을 binning하고 valid/test/model valid에  train 기준 binning 적용한 결과를 반환합니다.

    :param dataframe x_train_df: 독립변수 train 데이터
    :param dataframe x_valid_df: 독립변수 valid 데이터
    :param dataframe x_test_df: 독립변수 test 데이터
    :param dataframe x_model_valid_df: 독립변수 model valid 데이터
    :param list numeric_columns_list: 숫자형 변수 리스트
    :param array bin_cut: binning 기준 배열(numpy.linspace 권장)
    :return tuple: binning 된 train/valid/test/model valid dataframe
    """
    for variable in numeric_columns_list:
        # 각 numeric 항목을 등구간화
        x_train_df[variable] = pd.qcut(x=x_train_df[variable], q=bin_cut, duplicates='drop')
        value_len = len(x_train_df[variable].unique())
        categories_len = len(x_train_df[variable].cat.categories)

        # category 길이와 unique value 길이 비교
        # 다르다면 category에 0건인 구간이 있다고 판단 가능
        if value_len != categories_len:
            chk_cnt_df = pd.DataFrame(x_train_df[variable].value_counts().sort_index())
            chk_cnt_df = chk_cnt_df.reset_index(drop=False)
            chk_cnt_df.columns = ['value', 'cnt']
            zero_cnt = chk_cnt_df[chk_cnt_df['cnt'] == 0][['value', 'cnt']]
            zero_cnt_idx = zero_cnt.index.tolist()
            zero_cnt_value = zero_cnt['value'].tolist()

            for idx, value in zip(zero_cnt_idx, zero_cnt_value):
                # 0건인 bin의 order에 따른 interval 조정
                if idx == 0 or idx == len(chk_cnt_df) - 1:
                    x_train_df[variable].cat.remove_catgories(value, inplace=True)
                else:
                    lower_value = chk_cnt_df.loc[idx - 1, 'value']
                    upper_value = chk_cnt_df.loc[idx + 1, 'value']
                    change_upper_value = pd.Interval(left=lower_value.right, right=upper_value.right)
                    x_train_df[variable].cat.remove_categories(value, inplace=True)
                    x_train_df[variable].cat.rename_categories({upper_value: change_upper_value}, inplace=True)

        # 구간의 최소, 최대 구간의 극단값 조정
        # 최소 극단닶 -> -np.inf, 최대값 -> np.inf
        max_interval = x_train_df[variable].max()
        change_max_interval = pd.Interval(left=max_interval.left, right=np.inf)

        min_interval = x_train_df[variable].min()
        change_min_interval = pd.Interval(left=-np.inf, right=min_interval.right)

        # 최소, 최대 구간의 극단값 변경 적용
        x_train_df[variable] = x_train_df[variable].cat.rename_categories({max_interval: change_max_interval,
                                                                           min_interval: change_min_interval})

        # categorical 타입으로 변경
        x_train_df[variable] = pd.Categorical(x_train_df[variable])
        # train set 기준으로 구간화한 값을 valid, test, model_valid set에 똑같이 적용하기 위한 구간값 생성
        bin_points = x_train_df[variable].cat.categories
        # 구간화 적용
        x_valid_df[variable] = pd.cut(x=x_valid_df[variable], bins=bin_points, duplicates='drop')
        x_test_df[variable] = pd.cut(x=x_test_df[variable], bins=bin_points, duplicates='drop')
        x_model_valid_df[variable] = pd.cut(x=x_model_valid_df[variable], bins=bin_points, duplicates='drop')
    return x_train_df, x_valid_df, x_test_df, x_model_valid_df


def parallelize_fine_classing(x_train_df, x_valid_df, x_test_df, x_model_valid_df, numeric_columns_list, bin_cut, func,
                              processes):
    """
    :explain: 병렬처리를 적용하여 fine classing 결과를 반환합니다.

    :param dataframe x_train_df: 독립변수 train 데이터
    :param dataframe x_valid_df: 독립변수 valid 데이터
    :param dataframe x_test_df: 독립변수 test 데이터
    :param dataframe x_model_valid_df: 독립변수 model valid 데이터
    :param list numeric_columns_list: 숫자형 변수 리스트
    :param array bin_cut: binning 기준 배열(numpy.linspace 권장)
    :param function func: 병렬처리할 fine_classing 함수
    :param int processes: 병렬처리에 사용할 core 개수
    :return tuple:  binning 된 train/valid/test/model valid dataframe
    """
    # 병렬처리를 위해 numeric 항목 리스트를 processes(input)값 기준으로 n등분
    split_num_cols = np.array_split(numeric_columns_list, processes, axis=0)
    # 병렬처리 수행
    pool = mp.Pool(processes=processes)
    arg_tuple_list = [(x_train_df[split_col], x_valid_df[split_col], x_test_df[split_col],
                       x_model_valid_df[split_col], split_col, bin_cut)
                      for split_col in split_num_cols]
    result = pool.starmap(func, arg_tuple_list)
    pool.close()
    pool.join()

    x_train_df_list = []
    binned_val_x_list = []
    binned_tst_x_list = []
    binned_model_val_x_list = []
    # 병렬처리 결과값을 train, valid, test ,model_valid로 나눔
    for value in result:
        x_train_df_list.append(value[0])
        binned_val_x_list.append(value[1])
        binned_tst_x_list.append(value[2])
        binned_model_val_x_list.append(value[3])

    x_train_df = pd.concat(x_train_df_list, axis=1)
    binned_val_x = pd.concat(binned_val_x_list, axis=1)
    binned_tst_x = pd.concat(binned_tst_x_list, axis=1)
    binned_model_val_x = pd.concat(binned_model_val_x_list, axis=1)

    return x_train_df, binned_val_x, binned_tst_x, binned_model_val_x
