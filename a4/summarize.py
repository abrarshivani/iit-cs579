"""
sumarize.py
"""
import pickle

def get_cluster_summary(input):
    with open(input, "rb") as handle:
        number_of_users_collected = pickle.load(handle)
        number_of_communities_discovered = pickle.load(handle)
        average_number_of_users_per_community = pickle.load(handle)
    return number_of_users_collected, number_of_communities_discovered, average_number_of_users_per_community

def get_classify_summary(input):
    with open(input, "rb") as handle:
        number_of_messages = pickle.load(handle)
        number_of_instances_per_class_found = pickle.load(handle)
        class_examples = pickle.load(handle)
    return number_of_messages, number_of_instances_per_class_found, class_examples


def main():
    summary = open("summary.txt", "w")
    number_of_users_collected, number_of_communities_discovered, average_number_of_users_per_community = get_cluster_summary("cluster_summary")
    number_of_messages, number_of_instances_per_class_found, class_examples = get_classify_summary("classify_summary")
    summary.write("Number of users collected: %d\n" % number_of_users_collected)
    summary.write("Number of messages collected: %d\n" % number_of_messages)
    summary.write("Number of communities discovered: %d\n" % number_of_communities_discovered)
    summary.write("Average number of users per community: %d\n" % average_number_of_users_per_community)
    summary.write("Number of instances per class found: ")
    for label, instances in number_of_instances_per_class_found.items():
        summary.write(str(instances) + " ")
    summary.write("\nOne example from each class: \n")
    for label, examples in class_examples.items():
        summary.write("Class %s\n%s\n" % (label, examples))
    summary.close()

if __name__ == '__main__':
    main()