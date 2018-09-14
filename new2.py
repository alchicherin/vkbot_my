#задание 4
"""4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени
принадлежности положительному классу для каждого классификатора на некоторой выборке:

для логистической регрессии — вероятность положительного класса (колонка score_logreg),
для SVM — отступ от разделяющей поверхности (колонка score_svm),
для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
для решающего дерева — доля положительных объектов в листе (колонка score_tree).
Загрузите этот файл."""
from sklearn.metrics import precision_recall_curve
import pandas as pd
from collections import defaultdict

df=pd.read_csv("scores.csv")

"""5. Посчитайте площадь под ROC-кривой для каждого классификатора. 
Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? 
Воспользуйтесь функцией sklearn.metrics.roc_auc_score."""
import sklearn
from sklearn.metrics import roc_auc_score


def auc(clf_name:str):
    return roc_auc_score(df['true'], df[clf_name])


classificator_array = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']


defdict1 = defaultdict(list)
for i in classificator_array:
    defdict1[i].append(auc(i))

MAX0=max(defdict1.values())

for k0,v0 in defdict1.items():
    if v0 == MAX0:
        print("answer5=", k0)



"""6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?

Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции 
sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds. 
В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds. 
Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7."""




def max_precision(name_of_classificator):
    ARR_precision = []
    prec_arr, recall_arr, tres_arr = precision_recall_curve(df['true'],df[name_of_classificator])
    for i in range(0, len(tres_arr)):
        if recall_arr[i] >= 0.7:
            ARR_precision.append(prec_arr[i])
    return max(ARR_precision)


temp_dict = defaultdict(list)
for name in classificator_array:
    temp_dict[name].append(max_precision(name))
MAX = max(temp_dict.values())
for k, v in temp_dict.items():
    if v == MAX:
        print("answer6 =", k)
