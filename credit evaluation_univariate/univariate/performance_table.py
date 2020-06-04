import pandas as pd
import numpy as np


def base_performance_table(train_calc_stats_dict, valid_calc_stats_dict, test_calc_stats_dict, layout):
    """
    :explain: 입력 받은 train/valid/test의 statistics 계산결과를 layout과 병합하여 항목 정보를 생성하는 함수입니다.

    :param dictionary train_calc_stats_dict: train 데이터의 독립변수 performance 결과 
    :param dictionary valid_calc_stats_dict: valid 데이터의 독립변수 performance 결과
    :param dictionary test_calc_stats_dict: test 데이터의 독립변수 performance 결과
    :param dataframe layout: layout 데이터
    :return dataframe: 항목별 정보, performance값 결과
    """
    contents = {'Variable',
                'ks_trn', 'ks_val', 'ks_tst',
                'ar_trn', 'ar_val', 'ar_tst',
                'iv_trn', 'iv_val', 'iv_tst'}
    performance_df = pd.DataFrame(columns=contents)
    performance_df = performance_df[['Variable',
                                     'ks_trn', 'ks_val', 'ks_tst',
                                     'ar_trn', 'ar_val', 'ar_tst',
                                     'iv_trn', 'iv_val', 'iv_tst']]

    use_cols = list(train_calc_stats_dict.keys())
    for variable in use_cols:
        update_df = {'Variable': variable,
                     'ks_trn': train_calc_stats_dict[variable]['KS'],
                     'ks_val': valid_calc_stats_dict[variable]['KS'],
                     'ks_tst': test_calc_stats_dict[variable]['KS'],
                     'ar_trn': train_calc_stats_dict[variable]['AR'],
                     'ar_val': valid_calc_stats_dict[variable]['AR'],
                     'ar_tst': test_calc_stats_dict[variable]['AR'],
                     'iv_trn': train_calc_stats_dict[variable]['IV'],
                     'iv_val': valid_calc_stats_dict[variable]['IV'],
                     'iv_tst': test_calc_stats_dict[variable]['IV']}
        performance_df = performance_df.append(update_df, ignore_index=True)

    performance_df = performance_df.sort_values(by=['iv_trn'], ascending=False)
    result = layout.merge(performance_df, on=['Variable'], how='inner')
    result = result.set_index('Variable', drop=True, verify_integrity=True)
    result = result.sort_index()
    return result
