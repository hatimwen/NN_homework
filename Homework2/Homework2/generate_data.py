def generate_data_print(n):
    """
    Input: n represents the dimension of input data
    return: a list of input data
    """
    x_data = []
    for i in range(pow(2, n)):
        x = "{:0b}".format(i).zfill(n)
        x_data.append(x)
        # print(x)
    return x_data

def parity_check(x_data):
    y = []
    for x in x_data:
        y.append(x.count('1')%2)
    return y

if __name__ == "__main__":
    n = 8
    x_data = []
    x_data = generate_data_print(n)
    # print(x_data)
    y = parity_check(x_data)
    # print(y)
    x_y = []
    for i in range(pow(2, n)):
        xy = x_data[i]+str(y[i])
        print(xy)
        x_y.append(xy)
    # print(x_y)