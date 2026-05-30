import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

print("Shape of Dataset:")
print(iris.shape)
print("\nColumn Names:")
print(iris.columns)
print("\nFirst 5 Rows:")
print(iris.head())

iris.info()

print("\nDataset Information:")
print("\nSummary Statistics:")
print(iris.describe())
sns.set(style="whitegrid")

# display scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=iris,
    x='sepal_length',
    y='petal_length',
    hue='species'
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.show()

# display histograms
iris.hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# display box plot
plt.figure(figsize=(10,6))
sns.boxplot(data=iris)
plt.title("Box Plot of Iris Features")
plt.show()