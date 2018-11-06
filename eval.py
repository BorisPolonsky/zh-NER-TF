import os


def conlleval(label_predict, label_path, metric_path, delimiter="\t"):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :param delimiter
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                char = char.encode("utf-8")
                line.append("{}{delimiter}{}{delimiter}{}\n".format(char, tag, tag_, delimiter=delimiter))
            line.append("\n")
        fw.writelines(line)
    delimiter = delimiter.replace("\t", r'"\t"').replace(" ", '" "')
    if delimiter != " ":
        os.system("perl {} -o O -d {} < {} > {}".format(eval_perl, delimiter, label_path, metric_path))
    else:
        os.system("perl {} -o O < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics
