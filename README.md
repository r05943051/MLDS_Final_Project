--------------------------------------------------
Model形容： 使用Mnist data手寫辨識來實驗
主要test.py和visualize.py放在src資料夾裡面
--------------------------------------------------
執行方法 ： 
第一步驟 :   bash run_test.sh
就會開始分別使用tanh, elu, relu, leaky relu來做訓練
並將gradient存下來以便做gif檔
一定要做完第一步驟才能做第二個步驟
第二步驟:    bash run_vi.sh
就會開始在gif資料夾裡面產生gif圖來視覺話, 如果不想要自己做這件事, 
我們之前train好的結果已經有放在gif資料夾內可以參考
--------------------------------------------------
分為兩種model:
1. 單層神經網路→探討時間維度的微分傳遞
定義: 我們由單層神經網路去觀察每個時間點的輸入對loss函數的微分值，主要是觀察該激活函數在時間序列的增長時微分的變化情形如何。一開始我們有考慮使用1 * 784的序列作為長時間序列的測試，但發現在28個time step，微分傳遞的變化就已經很明顯了，因此為了模型簡易和統一化的考量，使用了28個time step的任務。
2. 多層神經網路→探討層與層之間的微分傳遞
定義: 我們由多層的神經網路去觀察每層的輸入對loss函數的微分值，因為28維的time step會造成圖像化的麻煩，且不是我們的主要觀察變因，因此我們將該維度取平均消除。此實驗主要是觀察該激活函數在層與層之前傳遞微分的能力。
---------------------------------------------------
參數設定:
圖col number： num_col = 28
圖row number： num_row = 28
pixel值：      num_px = num_col * num_row
分10類：       num_classes = 10
訓練資料：     num_examples = 60000
驗證資料：     num_examples_valiation = 10000
batch size = 32
學習率                   ：learning rate = 5e-5
epoch數                  ：number of epoch = 7
一個epoch有幾個batch      ：batch_per_epoch = num_examples / batch_size
每幾個step顯示一次結果    ：display_step = 50
每幾個step取一次gradient值：sample_gradient_step = 2
----------------------------------------------------

