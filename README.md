# 2021iFLYTEK_UserPortrait_Rank6
科大讯飞 基于用户画像的商品推荐挑战赛Top6
小菜鸡第一次参加的比赛，代码有些拉垮

赛题主要任务是对用户付费行为进行预测， 帮助进行更精准的产品推荐。

## 总结
初赛基于用户基本人口统计信息(类别变量)和用户标签信息(文本)等特征维度，在经典推荐算法 DeepFM 基础上，
结合多个文本分类模型(TextCNN、 TextBiRNN、 Capsule)， 处理用户标签信息， 用于预测用户是否购买相应商品；模型初赛 F1 得分 0.75176，Rank11。
复赛沿用初赛模型，采用伪标签方法，并进行多模型融合，最终 F1 得分 0.7681， Rank6。

## 参考
1、https://zhuanlan.zhihu.com/p/28923961

2、https://zhuanlan.zhihu.com/p/29020616

3、https://zhuanlan.zhihu.com/p/109933924

4、https://cloud.tencent.com/developer/article/1536537
