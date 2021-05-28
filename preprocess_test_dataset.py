with open("dataset.txt", "r") as f:
    with open("cleaned_dataset.txt", "w") as fw:
        write_lines = []
        for line in f:
            write_lines.append(line.strip("\n"))
            write_lines.append("<eos>")

        fw.write(" ".join(write_lines))
