#把文件读取为dataframe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import matplotlib.pyplot as plt

def load_data():
	# 从CSV文件加载数据
	df = pd.read_csv('all_stocks_5yr.csv')
	# 将"date_str"列转换为日期列
	df["date"] = pd.to_datetime(df["date"])
	# 将"date"列设置为索引列
	df.set_index("date", inplace=True)
	print(df)
	return df

#准备数据把y准备出来
# y是0或1（跌0涨1）
def process(df):

	# 计算涨跌情况
	df['change'] = df['close'].diff().shift(-1)
	# 添加涨跌标识列
	df['target'] = df['change'].apply(lambda x: 1 if x > 0 else 0)
	# 计算2日均值
	df['2_day_avg'] = df['close'].rolling(window=2).mean()
	# 计算10日均值
	df['10_day_avg'] = df['close'].rolling(window=10).mean()
	# 计算20日均值
	df['20_day_avg'] = df['close'].rolling(window=20).mean()
	# 计算价格的一阶差分
	df['dif1'] = df['close'].diff()
	# 计算价格的二阶差分
	df['dif2'] = df['dif1'].diff()
	# 计算价格的三阶差分
	df['dif3'] = df['dif2'].diff()

	# 计算2日均值
	df['volume_2_day_avg'] = df['volume'].rolling(window=2).mean()
	# 计算10日均值
	df['volume_10_day_avg'] = df['volume'].rolling(window=10).mean()
	# 计算20日均值
	df['volume_20_day_avg'] = df['volume'].rolling(window=20).mean()
	# 计算价格的一阶差分
	df['volume_dif1'] = df['volume'].diff()
	# 计算价格的二阶差分
	df['volume_dif2'] = df['volume_dif1'].diff()
	# 计算价格的三阶差分
	df['volume_dif3'] = df['volume_dif2'].diff()

	return df

def train():
	df = load_data()

	# 过滤掉appl

	# 假设要排除的股票代码为'AAPL'
	excluded_stock = 'AAPL'

	# 提取排除一只股票后的训练数据
	train_data = df[df['Name'] != excluded_stock]
	df = process(df)

	df = df.dropna()
	# 提取特征和标签
	X = df.drop(['target',"Name","change"], axis=1)  # 特征：除了涨跌标识列之外的所有列
	y = df['target']  # 标签：涨跌标识列

	# 划分训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 创建逻辑回归模型
	model = LogisticRegression()
	# 查看DataFrame的列
	print("x的特征值：",X_train.columns)
	# 拟合训练数据
	model.fit(X_train, y_train)

	# 在测试集上进行预测
	y_pred = model.predict(X_test)

	# 计算准确率
	accuracy = accuracy_score(y_test, y_pred)

	print(f"Accuracy: {accuracy}")

	# 保存模型
	dump(model, 'model.joblib')

	# 加载模型
	loaded_model = load('model.joblib')

def predict():
	df = load_data()

	# 提取苹果股票的数据
	df = df[df['Name'] == 'AAPL']

	df = process(df)
	df = df.dropna()
	# 提取特征和标签
	X = df.drop(['target',"Name","change"], axis=1)  # 特征：除了涨跌标识列之外的所有列
	y = df['target']  # 标签：涨跌标识列


	# 加载模型
	loaded_model = load('model.joblib')

	# 进行预测
	
	predictions = loaded_model.predict(X)

	# 打印预测结果
	print(predictions)
	# 计算准确率
	accuracy = accuracy_score(y, predictions)
	print(f"Accuracy: {accuracy}")

	df['new_target'] = predictions

	plot(df)


def plot(df):
	df = df.iloc[-200:]
	# 绘制折线图
	plt.plot(df.close)
	plt.xlabel('date')
	plt.ylabel('price')
	plt.title('stock:AAPL')
	# 绘制涨跌情况的红色三角
	plt.scatter(df[df.new_target == 1].index, df.close[df.new_target == 1], marker='v', color='red')
	plt.show()


if __name__ == '__main__':
	train()
	predict()
