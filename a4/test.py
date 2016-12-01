    with open(filename, "w") as handle:
        handle.write("Number of instances per class found: ")
        for label, instances in number_of_instances_per_class_found.items():
            handle.write(str(instances) + " ")
        handle.write("\nOne example from each class: \n")
        for label, examples in class_examples.items():
            handle.write("Class %s:- %s\n" % (label, examples))
