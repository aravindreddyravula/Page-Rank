from pyspark import SparkContext, SparkConf
import sys
import csv
from pprint import pprint
import numpy as np
from scipy import stats

def runTests(sc, word_data_by_county, heart_disease_data_by_county):
    def bonferroni_correction(top_words_list, count):
        return [(top_word[0], top_word[1][1], top_word[1][-2], top_word[1][-1] * count) for top_word in top_words_list]

    # Calculating p and t for the case 1 and 2
    def get_t_value_ht_w_hdm(beta, trainX, trainY):
        RSS = np.sum(np.power(trainY - beta[0] - (beta[1] * trainX), 2))
        length = len(trainX)
        df = length - len(beta)
        if df <= 0:
            return [0, 0]
        ss = RSS / df  # s_square
        t = beta[1] / (np.sqrt(ss / np.sum(np.power(trainX - np.mean(trainX), 2))))
        p = 2 * stats.t.sf(np.abs(t), length - 1)
        return [t, p]

    # Calculating p and t for the case 3 and 4
    def get_t_value_ht_w_hdm_cfm(beta, trainX, trainY):
        trainX = np.transpose(trainX)
        RSS = np.sum(np.power(trainY - beta[0] - (beta[1] * trainX[1]) - (beta[2] * trainX[2]), 2))
        length = len(trainX[0])
        df = length - len(beta)
        if df <= 0:
            return [0, 0]
        ss = RSS / df
        t = beta[1] / (np.sqrt(ss / np.sum(np.power(trainX[1] - np.mean(trainX[1]), 2))))
        p = 2 * stats.t.sf(np.abs(t), length - 1)
        return [t, p]

    # For the cases 1 and 2
    def ht_w_hdm(word, x):
        trainX = np.asarray([val[0] for val in x[1]])
        if np.std(trainX) == 0:
            return [0] * 4
        trainY = np.asarray([val[2] for val in x[1]])
        trainX = (trainX - np.mean(trainX)) / np.std(trainX)
        trainY = (trainY - np.mean(trainY)) / np.std(trainY)
        mean_X, mean_Y = np.mean(trainX), np.mean(trainY)
        sum_num = np.sum(np.multiply(trainX - mean_X, trainY - mean_Y))
        sum_den = np.sum(np.power(trainX - mean_X, 2))
        beta1 = sum_num / sum_den
        beta0 = mean_Y - beta1 * mean_X
        beta = [beta0, beta1]
        t = get_t_value_ht_w_hdm(beta, trainX, trainY)
        beta.extend(t)
        return beta

    # For the cases 3 and 4
    def ht_w_hdm_cfm(word, x):
        trainX1 = np.asarray([[val[0], val[1]] for val in x[1]])
        if np.any(np.std(trainX1, axis=0) == 0):
            return [0] * 5
        trainY = np.asarray([val[2] for val in x[1]])
        trainY = np.transpose(trainY)
        trainY = (trainY - np.mean(trainY, axis=0)) / np.std(trainY, axis=0)
        trainX1 = (trainX1 - np.mean(trainX1, axis=0)) / np.std(trainX1, axis=0)
        trainX2 = np.asarray([[1] for val in x[1]])
        trainX = np.concatenate((trainX2, trainX1), axis=1)
        beta = np.matmul(np.linalg.pinv(trainX), trainY)
        t = get_t_value_ht_w_hdm_cfm(beta, trainX, trainY)
        beta_list = list(beta)
        beta_list.extend(t)
        return beta_list

    def reduce_by_key_using_extend(x, y):
        x.extend(y)
        return x

    word_data_by_county_rdd = sc.textFile(word_data_by_county).mapPartitions(lambda line: csv.reader(line)).filter(
        lambda x: x[0] != "group_id")
    word_data_to_keyvalue = word_data_by_county_rdd.map(lambda x: (int(x[0]), [x[1], x[3]]))
    heart_disease_data_by_county_rdd = sc.textFile(heart_disease_data_by_county).mapPartitions(
        lambda line: csv.reader(line)).filter(lambda x: x[0] != "fips")
    heart_disease_data_required_rdd = heart_disease_data_by_county_rdd.map(lambda x: (int(x[0]), [x[23], x[24]]))
    output_rdd = word_data_to_keyvalue.join(heart_disease_data_required_rdd)
    output_rdd = output_rdd.map(lambda x: (
    x[1][0][0], [(float(x[1][0][1]), float(x[1][1][0]), float(x[1][1][1]))]))  # (word, [(rel_freq, log_median, heart)])
    output_rdd = output_rdd.reduceByKey(lambda a, b: reduce_by_key_using_extend(a, b))

    output_rdd_second = output_rdd
    count = output_rdd.count()

    output_rdd = output_rdd.map(lambda x: (x[0], ht_w_hdm(x[0], x)))
    print("##########################################################################")
    print("Case 1: The top 20 word positively correlated with heart disease mortality")
    print("##########################################################################")
    print("############FORMAT OF OUTPUT: [(WORD, BETA, T_VALUE, P_VALUE)]############")
    output = output_rdd.top(20, key=lambda x: x[1][1])
    output = bonferroni_correction(output, count)
    pprint(output)
    print("#########################################################################")
    print("Case 2: The top 20 word negatively correlated with heart disease mortality")
    print("##########################################################################")
    print("############FORMAT OF OUTPUT: [(WORD, BETA, T_VALUE, P_VALUE)]############")
    output = output_rdd.top(20, key=lambda x: -x[1][1])
    output = bonferroni_correction(output, count)
    pprint(output)

    output_rdd_second = output_rdd_second.map(lambda x: (x[0], ht_w_hdm_cfm(x[0], x)))
    print("######################################################################################")
    print("Case 3: The top 20 words positively related to heart mortality, controlling for income")
    print("######################################################################################")
    print("##################FORMAT OF OUTPUT: [(WORD, BETA, T_VALUE, P_VALUE)]##################")
    output = output_rdd_second.top(20, key=lambda x: x[1][1])
    output = bonferroni_correction(output, count)
    pprint(output)
    print("######################################################################################")
    print("Case 4: The top 20 words negatively related to heart mortality, controlling for income")
    print("######################################################################################")
    print("##################FORMAT OF OUTPUT: [(WORD, BETA, T_VALUE, P_VALUE)]##################")
    output = output_rdd_second.top(20, key=lambda x: -x[1][1])
    output = bonferroni_correction(output, count)
    pprint(output)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Invalid arguments")
        print(">spark-submit A2_P2_SparkHT_ravula_112026804.py word_data_file heart_disease_data_file --master yarn --deploy-mode cluster")
        sys.exit(0)
    word_data_by_county = sys.argv[1]
    heart_disease_data_by_county = sys.argv[2]
    conf = SparkConf().setAppName("aravind")
    sc = SparkContext(conf=conf)
    runTests(sc, word_data_by_county, heart_disease_data_by_county)
