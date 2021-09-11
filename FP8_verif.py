import os
import numpy as np
import matplotlib.pyplot as plt


class FloatEight:

    def __init__(self, num):
        self.F32 = np.float32(num)
        self.I32 = np.int32(0)
        self.F8 = np.float32(0)
        self.I8 = np.int32(0)
        self.f_I32()
        self.f_I8()
        self.f_F8()

    def f_I32(self):
        """decomposes a float32 into negative, exponent, and significand"""
        negative = self.F32 < 0
        n = np.abs(self.F32).view(np.int32)  # discard sign (MSB now 0),
        # view bit string as int32
        exponent = (n >> 23) - 127  # drop significand, correct exponent offset
        # 23 and 127 are specific to float32
        significand = n & np.int32(2 ** 23 - 1)  # second factor provides mask
        # to extract significand
        sign = format(negative, '01b')
        exp = format(exponent + 127, '08b')
        mant = format(significand, '023b')
        str_32b = sign + exp + mant
        self.I32 = int(str_32b, 2)
        self.H32 = format(self.I32, '08x')
        self.B32 = format(self.I32, '032b')

    def f_I8(self):
        num = abs(self.F32)
        num_dec = np.int32(num)
        num_frac = np.int32((num - num_dec) * (2 ** 16))
        combine = ['0'] * 32
        combine = format(num_dec, '016b') + format(num_frac, '016b')
        if combine.find('1') != -1:
            pos_of_one = combine.find('1')
            exp = -pos_of_one + 15
            biased_exp_num = exp + 15
            mant = combine[pos_of_one + 1:pos_of_one + 1 + 2]
            mant_num = int(mant, 2)

            # matissa normalization or rounding
            if combine[pos_of_one + 3] == '1':
                mant_num = mant_num + 1
                if mant_num == 4:
                    biased_exp_num = biased_exp_num + 1
                    mant_num = 0
            mant = format(mant_num, '02b')

            biased_exp = format(biased_exp_num, '05b')
            sign = format(self.F32 < 0, '01b')
        else:
            mant = format(0, '02b')
            biased_exp = format(0, '05b')
            sign = format(0, '01b')

        str_8b = sign + biased_exp + mant
        self.B8 = str_8b
        self.I8 = int(str_8b, 2)
        self.H8 = format(self.I8, '02x')
        return self.I8

    def f_F8(self):
        combine = format(self.I8, '08b')
        exp = int(combine[1:6], 2) - 15
        mant = '1' + combine[6:8]
        mant = int(mant, 2) / 4
        if self.I8 == 0:
            self.F8 = 0.0
        elif int(combine[0], 2):
            self.F8 = - mant * (2 ** exp)
        else:
            self.F8 = mant * (2 ** exp)

    def prt32b(self):
        print('----------------------------------')
        print('--------32 bit representation-----')
        print('----------------------------------')
        print('FLT | %f' % self.F32)
        print('BIN | %s_%s_%s' % (self.B32[0], self.B32[1:9], self.B32[9:32]))
        print('INT | %d' % self.I32)
        print('HEX | %s' % self.H32)

    def prt8b(self):
        print('----------------------------------')
        print('--------8 bit representation-----')
        print('----------------------------------')
        print('FLT | %f' % self.F8)
        print('BIN | %s_%s_%s : exp_int=%d or exp_hex=%x' % (
            self.B8[0], self.B8[1:6], self.B8[6:8], int(self.B8[1:6], 2), int(self.B8[1:6], 2)))
        print('INT | %d' % self.I8)
        print('HEX | %s' % self.H8)


def plot_the_curves():
    a = []
    b = []
    e = []
    N = 1000
    range_b = -10000
    range_a = 10000

    for i in range(0, N):
        num = (range_b - range_a) * np.random.random() + range_a
        a.append(num)
        b.append(FloatEight(num).F8)
        e.append(num - FloatEight(num).F8)
    # plt.plot(range(0, N), a)
    a.sort()
    b.sort()
    plt.plot(range(0, N), a, 'r')
    plt.plot(range(0, N), b, 'k')
    # plt.plot(range(0, N), e)
    print(max(e))
    idx = e.index(max(e))
    print(a[idx])
    print(b[idx])
    # plt.plot(range(0, N), b)
    plt.xlabel(' x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.show()


def int_to_F8(int_num):
    x = FloatEight(0)
    x.I8 = int_num
    x.f_F8()
    return x.F8


def hex_to_F8(hex_value):
    x = FloatEight(0)
    x.I8 = int(hex_value, 16)
    x.f_F8()
    return x.F8


def gen_test_vectors(file_name, opcode):
    file_name = os.path.join(os.getcwd(), 'test_vectors', file_name)
    range_b = -100
    range_a = 100
    f = open(file_name, 'w')
    for i in range(0, 1000):
        num1 = (range_b - range_a) * np.random.random() + range_a
        num2 = (range_b - range_a) * np.random.random() + range_a
        num1_F8 = FloatEight(num1)
        num2_F8 = FloatEight(num2)
        if opcode == 0:  # add
            golden_output = FloatEight(num1_F8.F8 + num2_F8.F8)
        else:  # mult
            golden_output = FloatEight(num1_F8.F8 * num2_F8.F8)
        str_to_write = num1_F8.H8 + num2_F8.H8 + golden_output.H8
        f.writelines("%s\n" % str_to_write)
    f.close()


def gen_tenosr_block_vectors(file_name, PDOT=8):
    file_name = os.path.join(os.getcwd(), 'test_vectors', file_name)
    range_b = -100
    range_a = 100
    f = open(file_name, 'w')
    for i in range(0, 100):
        psum = 0.0
        str_to_write = ''
        for j in range(0, PDOT):
            num1 = (range_b - range_a) * np.random.random() + range_a
            num2 = (range_b - range_a) * np.random.random() + range_a
            num1_F8 = FloatEight(num1)
            num2_F8 = FloatEight(num2)
            prod_F8 = FloatEight(num1_F8.F8 * num2_F8.F8)

            psum = psum + prod_F8.F8
            str_to_write = str_to_write + num1_F8.H8 + num2_F8.H8
            # print("%f,%f,%f,%f "%(num1_F8.F8,num2_F8.F8,prod_F8.F8,psum))
        str_to_write = FloatEight(psum).H8 + '_' + str_to_write
        f.writelines("%s\n" % str_to_write)
    f.close()


def verif_tensor_print():
    psum1 = int_to_F8(228)
    psum2 = int_to_F8(240)
    psum3 = int_to_F8(106)
    psum4 = int_to_F8(226)
    psum5 = int_to_F8(236)
    psum6 = int_to_F8(225)
    psum7 = int_to_F8(101)
    psum8 = int_to_F8(100)
    psum9 = int_to_F8(251)
    
    # print(FloatEight(psum1).B8)
    # print(FloatEight(psum2).B8)
    # print(FloatEight(psum3).B8)
    # print(FloatEight(psum4).B8)
    # print(FloatEight(psum5).B8)
    # print(FloatEight(psum6).B8)
    # print(FloatEight(psum7).B8)
    # print(FloatEight(psum8).B8)
    # print(FloatEight(psum9).B8)

    FloatEight(psum1).prt8b()
    FloatEight(psum2).prt8b()
    FloatEight(psum3).prt8b()
    FloatEight(psum4).prt8b()
    FloatEight(psum5).prt8b()
    FloatEight(psum6).prt8b()
    FloatEight(psum7).prt8b()
    FloatEight(psum8).prt8b()
    FloatEight(psum9).prt8b()

    sum = FloatEight(psum1 + psum2 + psum3 + psum4 + psum5 + psum6 + psum7 + psum8 + psum9)
    print(sum.H8)


def main():
    a = 0.5
    # b = 1.5
    # a = int_to_F8(245)
    # b = int_to_F8(203)
    # a = hex_to_F8('FC')
    # b = hex_to_F8('f4')
    # c = a + b
    FloatEight(a).prt8b()
    # FloatEight(b).prt8b()
    # FloatEight(c).prt8b()

    # print(hex_to_F8('46'))
    # print(hex_to_F8('53'))
    # print(hex_to_F8('54'))
    # print(hex_to_F8('84'))

    # gen_test_vectors('add_test.txt', opcode=0)
    # gen_test_vectors('mult_test.txt', opcode=1)
    np.random.seed(0)
    # gen_tenosr_block_vectors('tensor_test.txt', PDOT=8)
    # verif_tensor_print()

    # convolutional layer
    # for ii in range(5):
    #     tensor = np.load(f"./params/WU/conv_w{ii}_updated.npy")
    #     # import pdb;pdb.set_trace()
    #     holder = np.zeros(tensor.shape, dtype=object)
    #     for f in range(tensor.shape[0]):
    #         for c in range(tensor.shape[1]):
    #             for kh in range(tensor.shape[2]):
    #                 for kw in range(tensor.shape[3]):
    #                     w = tensor[f,c,kh,kw]
    #                     if abs(w) < 0.000125:
    #                         print(w)
    #                         w = 0.
    #                     hex_val = FloatEight(w).H8
    #                     holder[f,c,kh,kw] = hex_val
    #     # import pdb;pdb.set_trace()
    #     np.save(f"./params/WU/hex/conv_w{ii}_updated_hex.npy", holder)


    # fully connected layer
    tensor = np.load(f"./params/WU/fc_w5_updated.npy")
    tensor = tensor.reshape(1, -1)
    holder = np.zeros(tensor.shape, dtype=object)
    for f in range(tensor.shape[0]):
        for c in range(tensor.shape[1]):
            w = tensor[f,c]
            if abs(w) < 0.000125:
                print(w)
                w = 0.
            hex_val = FloatEight(w).H8
            holder[f,c] = hex_val
    np.save("./params/WU/hex/fc_w5_updated.npy", holder)
    print(tensor.shape)

    # # fc layer bias
    # tensor = np.load("./intermediate_results/FF/fc_b.npy")
    # holder = np.zeros(tensor.shape, dtype=object)
    # for f in range(tensor.shape[0]):
    #     w = tensor[f]
    #     if abs(w) < 0.000125:
    #         print(w)
    #         w = 0.
    #     hex_val = FloatEight(w).H8
    #     holder[f] = hex_val
    # np.save("./intermediate_results/FF/hex/fc_b_hex.npy", holder)

if __name__ == '__main__':
    main()
