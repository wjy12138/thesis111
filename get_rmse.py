import csv

file = 'result/figure_csv/20260329_100521_lb96_lf1-2-3-4-5-6/site1_lookforward_metrics_lb96.csv'
data = []
with open(file, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        data.append(row)

for lf in range(1, 7):
    lf_str = str(lf)
    lstm_rmse = float([d['rmse'] for d in data if d['model_name']=='LSTM' and d['look_forward']==lf_str][0])
    cnn_rmse = float([d['rmse'] for d in data if d['model_name']=='CNN' and d['look_forward']==lf_str][0])
    cnnlstm_rmse = float([d['rmse'] for d in data if d['model_name']=='CNN_LSTM' and d['look_forward']==lf_str][0])
    
    print(f'LF={lf}: LSTM={lstm_rmse:.3f}, CNN={cnn_rmse:.3f} ({(lstm_rmse-cnn_rmse)/lstm_rmse*100:.1f}%), CNN_LSTM={cnnlstm_rmse:.3f} ({(lstm_rmse-cnnlstm_rmse)/lstm_rmse*100:.1f}%)')
