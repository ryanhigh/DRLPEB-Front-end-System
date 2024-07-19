import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify, render_template, make_response
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Grid, Pie, Tab
from flask.json import jsonify
from random import randrange
from pyecharts.commons.utils import JsCode
import pyecharts

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__, template_folder="pages", static_folder="static")

def data_clean(dataframe):
    dataframe.replace('dataread', np.nan, inplace=True)
    dataframe_fillna = dataframe.fillna(method='ffill')
    # 将最后一列的区块传播事件暂时改为0
    dataframe_fillna['blockpropagationtime(ms)'] = 0

    # 查看info得知如下
    #   Column                    Non-Null Count  Dtype  
    # ---  ------                    --------------  -----  
    #  0   period(s)                 1161 non-null   int64  
    #  1   gaslimit                  1161 non-null   int64  
    #  2   tps(tx/s)                 1161 non-null   float64
    #  3   latency(ms)               1161 non-null   float64
    #  4   contracttime(μs)          1161 non-null   float64
    #  5   dbreaadtime(μs)           1161 non-null   float64
    #  6   dbwritetime(μs)           1161 non-null   float64
    #  7   blockcommittime(ms)       1161 non-null   float64
    #  8   readtime(ms)              1161 non-null   object 
    #  9   blockpropagationtime(ms)  1161 non-null   int64  
    #  dtypes: float64(6), int64(3), object(1)
    #  故需要将readtime(ms)改为数字类型
    modify_func = lambda x: float(x[:-2])
    dataframe_fillna['readtime(ms)'] = dataframe_fillna['readtime(ms)'].apply(modify_func)
    return dataframe_fillna

# 返回dataframe的下一行的第m、n列数据
def next_xy(df,i,m,n):
    x = df.iloc[i+1,m]
    y = df.iloc[i+1,n]
    i = i+1
    return i,x,y
    
# 根据all_df的第0列和第n列的前2行数据，初始化一条line，
def line_base(df,line_name,n):
    # print('rows data:',df.iloc[:,0])
    # global ddqn_r2_idx
    line = (
        Line()
        .add_xaxis(xaxis_data=df.iloc[:2,0].tolist())#第0列是时间
        .add_yaxis(series_name=line_name,   
                   y_axis=df.iloc[:2,n].tolist(),
                   is_smooth=True, 
                   
                #    markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="min"),opts.MarkPointItem(type_="max")]),
                   )
        .set_global_opts(legend_opts=opts.LegendOpts(pos_left="6%",pos_top="0",
                                                     textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8")),
                         xaxis_opts=(opts.AxisOpts(type_="value",name='episodes',name_location='center',min_='dataMin',name_gap=8,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),
                                                   axislabel_opts=opts.LabelOpts(margin=8, font_size=8))),
                         yaxis_opts=(opts.AxisOpts(type_="value",name="reward",name_location='center',min_='dataMin',name_gap=10,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),axislabel_opts=opts.LabelOpts(margin=8, font_size=8))))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    return line


all_df = pd.read_csv('source/result2.csv')
all_clean_df = data_clean(all_df)

reward2_df = pd.read_csv('static/rewards.csv')
reward2_df = reward2_df.iloc[:, 1:] # tlr version

reward1_df = pd.read_csv('static/reward1s.csv')
reward1_df = reward1_df.round(3)
reward1_df = reward1_df.iloc[:, 1:] # complicted version

val_1_df = pd.read_csv('static/validateResult.csv')
val_1_df = val_1_df.round(3)
val_1_df = val_1_df.iloc[:, 1:]
group_1_idx = val_1_df.shape[0]-1

val_10_df = pd.read_csv('static/validate10groupResult.csv')
val_10_df = val_10_df.round(3)
val_10_df = val_10_df.iloc[:, 1:]
group_10_idx = val_10_df.shape[0]-1

@app.route('/')
def index():
    return render_template("drlpeb.html")

@app.route('/get_data', methods=['GET'])
def get_data():
    key1 = int(request.args.get('key1'))
    key2 = int(request.args.get('key2'))
    if key1 and key2:
        filtered_df = all_clean_df[(all_clean_df['gaslimit'] == key1) & (all_clean_df['period(s)'] == key2)]
        filtered_df = filtered_df.iloc[:, 2:]
        print(filtered_df)
        result = filtered_df.to_dict(orient='records')
        print(result)
        return jsonify(result)
    else:
        return jsonify([])

####################################### ddqn_rewardv2 chart ##################################
@app.route("/ddqn_rewardv2")
def ddqn_r2():
    l = line_base(reward2_df, 'DDQN reward', 2)
    return l.dump_options_with_quotes()

ddqn_r2_idx = 3 # reward2_df.shape[0]-1
@app.route("/ddqn_rewardv2_dynamicdata")
def ddqn_r2_dynamicdata():
    global ddqn_r2_idx
    if ddqn_r2_idx == reward2_df.shape[0]-1:
        return jsonify({"x_data": '', "y_data": ''})
    else:
        ddqn_r2_idx, x, y = next_xy(reward2_df, ddqn_r2_idx, 0, 2)
        # 创建数据字典
        data = {"x_data": x, "y_data": y}
        # 使用自定义的NumpyEncoder进行JSON序列化
        json_data = json.dumps(data, cls=NumpyEncoder)
        # 创建响应并设置内容类型为application/json
        response = make_response(json_data)
        response.headers['Content-Type'] = 'application/json'
        return response
    
####################################### ddqn_rewardv1 chart ##################################
@app.route("/ddqn_rewardv1")
def ddqn_r1():
    l = line_base(reward1_df, 'DDQN reward', 2)
    return l.dump_options_with_quotes()

ddqn_r1_idx = 3 # reward1_df.shape[0]-1
@app.route("/ddqn_rewardv1_dynamicdata")
def ddqn_r1_dynamicdata():
    global ddqn_r1_idx
    if ddqn_r1_idx == reward1_df.shape[0]-1:
        return jsonify({"x_data": '', "y_data": ''})
    else:
        ddqn_r1_idx, x, y = next_xy(reward1_df, ddqn_r1_idx, 0, 2)
        # 创建数据字典
        data = {"x_data": x, "y_data": y}
        # 使用自定义的NumpyEncoder进行JSON序列化
        json_data = json.dumps(data, cls=NumpyEncoder)
        # 创建响应并设置内容类型为application/json
        response = make_response(json_data)
        response.headers['Content-Type'] = 'application/json'
        return response

####################################### ppo_rewardv2 chart ##################################
@app.route("/ppo_rewardv2")
def ppo_r2():
    l = line_base(reward2_df, 'PPO reward', 3)
    return l.dump_options_with_quotes()

ppo_r2_idx = 3 # reward2_df.shape[0]-1
@app.route("/ppo_rewardv2_dynamicdata")
def ppo_r2_dynamicdata():
    global ppo_r2_idx
    if ppo_r2_idx == reward2_df.shape[0]-1:
        return jsonify({"x_data": '', "y_data": ''})
    else:
        ppo_r2_idx, x, y = next_xy(reward2_df, ppo_r2_idx, 0, 3)
        # 创建数据字典
        data = {"x_data": x, "y_data": y}
        # 使用自定义的NumpyEncoder进行JSON序列化
        json_data = json.dumps(data, cls=NumpyEncoder)
        # 创建响应并设置内容类型为application/json
        response = make_response(json_data)
        response.headers['Content-Type'] = 'application/json'
        return response

####################################### ppo_rewardv1 chart ##################################
@app.route("/ppo_rewardv1")
def ppo_r1():
    l = line_base(reward1_df, 'PPO reward', 3)
    return l.dump_options_with_quotes()

ppo_r1_idx = 3 # reward1_df.shape[0]-1
@app.route("/ppo_rewardv1_dynamicdata")
def ppo_r1_dynamicdata():
    global ppo_r1_idx
    if ppo_r1_idx == reward1_df.shape[0]-1:
        return jsonify({"x_data": '', "y_data": ''})
    else:
        ppo_r1_idx, x, y = next_xy(reward1_df, ppo_r1_idx, 0, 3)
        # 创建数据字典
        data = {"x_data": x, "y_data": y}
        # 使用自定义的NumpyEncoder进行JSON序列化
        json_data = json.dumps(data, cls=NumpyEncoder)
        # 创建响应并设置内容类型为application/json
        response = make_response(json_data)
        response.headers['Content-Type'] = 'application/json'
        return response
    
####################################### val chart ##################################
def line_all(df, line_name, n, group_idx, y_name):
    line = (
        Line()
        .add_xaxis(xaxis_data=df.iloc[:group_idx,0].tolist())#第0列是时间
        .add_yaxis(series_name=line_name,   
                   y_axis=df.iloc[:group_idx,n].tolist(),
                   is_smooth=True, 
                   
                #    markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="min"),opts.MarkPointItem(type_="max")]),
                   )
        .set_global_opts(legend_opts=opts.LegendOpts(pos_left="6%",pos_top="0",
                                                     textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8")),
                         xaxis_opts=(opts.AxisOpts(type_="value",name='episodes',name_location='center',min_='dataMin',name_gap=15,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),
                                                   axislabel_opts=opts.LabelOpts(margin=8, font_size=8))),
                         yaxis_opts=(opts.AxisOpts(type_="value",name=y_name,name_location='center',min_='dataMin',name_gap=30,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),axislabel_opts=opts.LabelOpts(margin=8, font_size=8))))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    return line

@app.route("/ddqn_1_tps")
def ddqn_group1_tps():
    l = line_all(val_1_df, 'DDQN_TPS', 1, group_1_idx, "TPS")
    return l.dump_options_with_quotes()

@app.route("/ddqn_1_delay")
def ddqn_group1_delay():
    l = line_all(val_1_df, 'DDQN_Latency', 2, group_1_idx, "Latency")
    return l.dump_options_with_quotes()

@app.route("/ppo_1_tps")
def ppo_group1_tps():
    l = line_all(val_1_df, 'PPO_TPS', 3, group_1_idx, "TPS")
    return l.dump_options_with_quotes()

@app.route("/ppo_1_delay")
def ppo_group1_delay():
    l = line_all(val_1_df, 'PPO_Latency', 4, group_1_idx, "Latency")
    return l.dump_options_with_quotes()

@app.route("/10_tps")
def group10_tps():
    line = (
        Line()
        .add_xaxis(xaxis_data=val_10_df.iloc[:group_10_idx,0].tolist())#第0列是时间
        .set_global_opts(legend_opts=opts.LegendOpts(pos_left="6%",pos_top="0",
                                                     textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8")),
                         xaxis_opts=(opts.AxisOpts(type_="value",name='episodes',name_location='center',min_='dataMin',name_gap=15,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),
                                                   axislabel_opts=opts.LabelOpts(margin=8, font_size=8))),
                         yaxis_opts=(opts.AxisOpts(type_="value",name="TPS",name_location='center',min_='dataMin',name_gap=30,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),axislabel_opts=opts.LabelOpts(margin=8, font_size=8))))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    for i in range(10):
        # 使用不同的颜色
        color = f'#{i:02x}{(i*10):02x}{(i*20):02x}'
        line.add_yaxis(series_name='group{}'.format(i+1),   
                   y_axis=val_10_df.iloc[:group_10_idx,2*(i+1)-1].tolist(),
                   is_smooth=True, 
                   itemstyle_opts=opts.ItemStyleOpts(color=color)
                   )
        line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    return line.dump_options_with_quotes()

@app.route("/10_delay")
def group10_delay():
    line = (
        Line()
        .add_xaxis(xaxis_data=val_10_df.iloc[:group_10_idx,0].tolist())#第0列是时间
        .set_global_opts(legend_opts=opts.LegendOpts(pos_left="6%",pos_top="0",
                                                     textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8")),
                         xaxis_opts=(opts.AxisOpts(type_="value",name='episodes',name_location='center',min_='dataMin',name_gap=15,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),
                                                   axislabel_opts=opts.LabelOpts(margin=8, font_size=8))),
                         yaxis_opts=(opts.AxisOpts(type_="value",name="Latency",name_location='center',min_='dataMin',name_gap=30,
                                                   name_textstyle_opts=opts.TextStyleOpts(font_family="微软雅黑",font_size="8"),axislabel_opts=opts.LabelOpts(margin=8, font_size=8))))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    for i in range(10):
        # 使用不同的颜色
        color = f'#{i:02x}{(i*10):02x}{(i*20):02x}'
        line.add_yaxis(series_name='group{}'.format(i+1),   
                   y_axis=val_10_df.iloc[:group_10_idx,2*(i+1)].tolist(),
                   is_smooth=True, 
                   itemstyle_opts=opts.ItemStyleOpts(color=color)
                   )
        line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    return line.dump_options_with_quotes()




if __name__ == '__main__':
    # print(all_clean_df.info())
    app.run(debug=True)
