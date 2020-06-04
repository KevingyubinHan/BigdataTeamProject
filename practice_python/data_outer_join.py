# data_outer_join.py

import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

# load data
data_dir = r'D:\Classification Rule\Regular Expression\04. 데이터그룹화 추가작업\200529_오락실,테마파크,아쿠아리움,워터파크'
data_path = os.path.join(data_dir, '정보분류_작업중.xlsx')
data_sheetname = 'Layout'
classified_data_df = pd.read_excel(io=data_path, sheet_name=data_sheetname, keep_default_na=False)

hit_dir = r'D:\Classification Rule\Regular Expression\03. 정규식 match 결과'
hit_path = os.path.join(hit_dir, 'RegExpr_hit_total_200529_오락실,테마파크,아쿠아리움,워터파크(전체)_4차.xlsx')
hit_total_df = pd.read_excel(io=hit_path)

# 전처리
classified_data_df['chk_multiple'] = classified_data_df['업체명'].str.contains(',')
classified_data_df['chk_multiple'] = classified_data_df['chk_multiple'].map({True: 1, False: 0})
classified_data_df = classified_data_df[classified_data_df['chk_multiple'] == 0]

data_need_cols = ['detail', '적요형태', 'CODE']
classified_data_df = classified_data_df[classified_data_df['완료여부'] == '완료'][data_need_cols]
classified_data_df.sort_values(by=['CODE', 'detail'], inplace=True, ascending=True)
classified_data_df.drop_duplicates(subset='detail', keep='first', inplace=True)
classified_data_df.reset_index(drop=True, inplace=True)

#hit_total_df.sort_values(by=['rule_index', 'desc'], inplace=True, ascending=True)
#hit_total_df.drop_duplicates(subset='desc', keep='first', inplace=True)
#hit_total_df.to_excel(hit_path) #중복제거 저장

hit_need_cols = ['rule_index', 'desc', 'info_code', 'region1', 'region2', 'region3', 'region4', 'cond_desc']
hit_total_df = hit_total_df[hit_need_cols]
hit_total_df.sort_values(by=['info_code', 'desc'], inplace=True, ascending=True)
hit_total_df.reset_index(drop=True, inplace=True)

# check record (detail-desc)
classified_data_df['chk_record'] = classified_data_df['detail'].isin(hit_total_df['desc'].tolist())
classified_data_df['chk_record'] = classified_data_df['chk_record'].map({True: 1, False: 0})
hit_total_df['chk_record'] = hit_total_df['desc'].isin(classified_data_df['detail'].tolist())
hit_total_df['chk_record'] = hit_total_df['chk_record'].map({True: 1, False: 0})

# 미분류 label 저장 (df)
detail_not_in_desc = classified_data_df[classified_data_df['chk_record'] == 0]
desc_not_in_detail = hit_total_df[hit_total_df['chk_record'] == 0]
print('detail_not_in_desc: ', detail_not_in_desc.shape[0])
print('desc_not_in_detail: ', desc_not_in_detail.shape[0])

if detail_not_in_desc.shape[0] > 0:
    detail_not_in_desc.to_excel(data_dir + os.sep + 'detail_not_in_desc.xlsx')
if desc_not_in_detail.shape[0] > 0:
    desc_not_in_detail.to_excel(data_dir + os.sep + 'desc_not_in_detail.xlsx')

# outer join (오분류 label 포함)
classified_data_df = classified_data_df[classified_data_df['chk_record'] == 1].drop(columns='chk_record')
classified_data_df.sort_values(by=['detail'], inplace=True, ascending=True)
classified_data_df.reset_index(drop=True, inplace=True)
hit_total_df = hit_total_df[hit_total_df['chk_record'] == 1].drop(columns='chk_record')
hit_total_df.sort_values(by=['desc'], inplace=True, ascending=True)
hit_total_df.reset_index(drop=True, inplace=True)

outer_join_df = pd.concat(objs=[classified_data_df, hit_total_df], axis=1, join='outer', ignore_index=True)
outer_join_df.columns = data_need_cols + hit_need_cols


# check code (CODE-info_code)
def check_code(CODE, info_code):
    result = 0
    if CODE != info_code:
        result = 1
    return result


code_cols = ['CODE', 'info_code']
outer_join_df['chk_code'] = outer_join_df[code_cols].apply(lambda x: check_code(x['CODE'], x['info_code']), axis=1)
print('chk_code: ', outer_join_df['chk_code'].sum())


# check region (적요형태-region)
outer_join_df['region_in_record'] = outer_join_df['적요형태'].str.contains('\[지점명\]')
outer_join_df['region_in_record'] = outer_join_df['region_in_record'].map({True: 1, False: 0})


def contain_region(*args):
    result = 0
    for arg in args:
        if arg is not np.nan:
            result += 1
    return result


region_cols = ['region1', 'region2', 'region3', 'region4']
outer_join_df['contain_region'] = outer_join_df[region_cols].apply(
    lambda x: contain_region(x['region1'], x['region2'], x['region3'], x['region4']), axis=1)


def check_region(region_in_record, contain_region):
    result = 0
    if region_in_record > 0 and contain_region == 0:
        result = -1
    elif region_in_record == 0 and contain_region > 0:
        result = 1
    return result


need_cols = ['region_in_record', 'contain_region']
outer_join_df['chk_region'] = outer_join_df[need_cols].apply(
    lambda x: check_region(x['region_in_record'], x['contain_region']), axis=1)
print('chk_region -1: ', len(outer_join_df[outer_join_df['chk_region'] == -1]))
print('chk_region 1: ', len(outer_join_df[outer_join_df['chk_region'] == 1]))

outer_join_df.sort_values(by=['rule_index'], inplace=True)
outer_join_df.to_excel(data_dir + os.sep + 'outer_join.xlsx')