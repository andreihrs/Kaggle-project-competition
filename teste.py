from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(data_train, tl.ravel(), test_size=0.20)



from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



def generate_results(model, df_test, features, id_col, target, file):
    dft = df_test[features]
    results = df_test[[id_col]]
    results[target] = model.predict_proba(dft)[:, 1]
    results.to_csv(file, index=False, columns=results.columns)

    # split train and test data with function defaults
    # random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
    train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target],
                                                                            random_state=0)
    train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin],
                                                                                            data1[Target],
                                                                                            random_state=0)
    train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
        data1_dummy[data1_x_dummy], data1[Target], random_state=0)

    print("Data1 Shape: {}".format(data1.shape))
    print("Train1 Shape: {}".format(train1_x.shape))
    print("Test1 Shape: {}".format(test1_x.shape))

    train1_x_bin.head()

    # Set the width and height of the figure
    plt.figure(figsize=(16, 6))

    # Line chart showing how FIFA rankings evolved over time
    sns.lineplot(data=fifa_data)

    # Bar chart showing average arrival delay for Spirit Airlines flights by month
    sns.barplot(x=flight_data.index, y=flight_data['NK'])

    # Add label for vertical axis
    plt.ylabel("Arrival delay (in minutes)")

    # Heatmap showing average arrival delay for each airline by month
    sns.heatmap(data=flight_data, annot=True)

    sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

    #Verify the strength of the relationship
    sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

    sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])

    #Two regression lines
    sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

    sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

    # Histogram
    sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)

    # KDE plot
    sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)

    # 2D KDE plot
    sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")


# df.quality.value_counts()
#
# sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap = 'viridis') - to check missing values
#
# plt.figure(figsize=(6,4))
# sns.heatmap(df.corr(),cmap='Blues',annot=False)  - to check correlation
#
# #Quality correlation matrix
# k = 12 #number of variables for heatmap
# cols = df.corr().nlargest(k, 'quality')['quality'].index
# cm = df[cols].corr()
# plt.figure(figsize=(10,6))
# sns.heatmap(cm, annot=True, cmap = 'viridis')
#
# l = df.columns.values
# number_of_columns=12
# number_of_rows = len(l)-1/number_of_columns
# plt.figure(figsize=(number_of_columns,5*number_of_rows))
# for i in range(0,len(l)):
#     plt.subplot(number_of_rows + 1,number_of_columns,i+1)
#     sns.set_style('whitegrid')
#     sns.boxplot(df[l[i]],color='green',orient='v')
#     plt.tight_layout() - to check outliers
#
#     plt.figure(figsize=(2 * number_of_columns, 5 * number_of_rows))
#     for i in range(0, len(l)):
#         plt.subplot(number_of_rows + 1, number_of_columns, i + 1)
#         sns.distplot(df[l[i]], kde=True) - to check distribution skewness



    # generate_results(model, df_test, features, "SK_ID_CURR", "TARGET", "results/results.csv")

# !kaggle  competitions  submit -c home-credit-default-risk -f results.csv -m "Xgb baseline with numeric cols"

# plt.figure(figsize=(16, 6))
# sns.lineplot(data=df_train)
# plt.figure(figsize=(16, 6))
# sns.barplot(x=df_train.iloc[:,0], y=df_train.iloc[:,0])
# # sns.heatmap(data=df_train, annot=True)

# df_train.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
#            xlabelsize=8, ylabelsize=8, grid=False)
# plt.tight_layout(rect=(0, 0, 1.2, 1.2))

# fig = plt.figure(figsize = (6,4))
# title = fig.suptitle("X coordinates", fontsize=14)
# fig.subplots_adjust(top=0.85, wspace=0.3)

# ax = fig.add_subplot(1,1, 1)
# ax.set_xlabel("X")
# ax.set_ylabel("Value field")
# ax.text(1.2, 10, r'$\mu$='+str(round(df_train.iloc[:,-1].mean(),2)),
#          fontsize=12)
# freq, bins, patches = ax.hist(df_train.iloc[:,-1], color='steelblue', bins=15,
#                                     edgecolor='black', linewidth=1)


# fig = plt.figure(figsize = (6, 4))
# fig = plt.figure(figsize = (6, 4))
# title = fig.suptitle("X coordinates", fontsize=14)
# fig.subplots_adjust(top=0.85, wspace=0.3)

# ax1 = fig.add_subplot(1,1, 1)
# ax1.set_xlabel("X")
# ax1.set_ylabel("Value field")
# sns.kdeplot(df_train.iloc[:,-1], ax=ax1, shade=True, color='steelblue')

# Correlation Matrix Heatmap
# f, ax = plt.subplots(figsize=(10, 6))
# corr = df_train.corr()
# hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
#                  linewidths=.05)
# f.subplots_adjust(top=0.93)
# t= f.suptitle('Coordinates Correlation Heatmap', fontsize=14)