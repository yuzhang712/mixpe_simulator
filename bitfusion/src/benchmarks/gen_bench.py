import pickle
import random
# gen cycle for standard opt, int8*int8, fp16*fp16
# much divide batch_size during dequant to obtain accurate cycle num
def gen_opt(n_layers, hidden_states, precision_bits, batch_size):
    opt = []
    # f = open(file_path, 'rb')
    # data = pickle.load(f)
    for i in range(n_layers):
        qkv_bit = dot_bit = ffn1_bit = ffn2_bit = precision_bits[1]
        qkv_dequant_bit = dot_dequant_bit = ffn1_dequant_bit = ffn2_dequant_bit = precision_bits[1]
        # qkv dequant(element-wise product)
        qkv_dequant = [[1, hidden_states] , [int(hidden_states*3/batch_size), hidden_states] , [1, int(hidden_states*3/batch_size)] , [] , [],  qkv_dequant_bit , 1]
        qkv = [[1, hidden_states] , [hidden_states*3, hidden_states] , [1, hidden_states*3] , [] , [],  qkv_bit , 1]
        
        # output projection dequant(element-wise product)
        dot_product_dequant = [[1, hidden_states] , [int(hidden_states/batch_size), hidden_states] , [1, int(hidden_states/batch_size)] , [] , [],  dot_dequant_bit , 1]
        dot_product = [[1, hidden_states] , [hidden_states, hidden_states] , [1, hidden_states] , [] , [],  dot_bit , 1]
        
        # ffn1 dequant
        ffn_1_dequant = [[1, hidden_states] , [int(hidden_states*4/batch_size), hidden_states] , [1, int(hidden_states*4/batch_size)] , [] , [],  ffn1_dequant_bit , 1]
        ffn_1 = [[1, hidden_states] , [hidden_states*4, hidden_states] , [1, hidden_states*4] , [] , [],  ffn1_bit , 1]
        
        # ffn2 dequant
        ffn_2_dequant = [[1, hidden_states*4] , [int(hidden_states/batch_size), hidden_states*4] , [1, int(hidden_states/batch_size)] , [] , [],  ffn2_dequant_bit , 1]
        ffn_2 = [[1, hidden_states*4] , [hidden_states, hidden_states*4] , [1, hidden_states] , [] , [],  ffn2_bit , 1]
        
        opt.extend((qkv_dequant, qkv, dot_product_dequant, dot_product, ffn_1_dequant, ffn_1, ffn_2_dequant, ffn_2))

    return opt

def gen_opt_mix(n_layers, hidden_states, precision_bits, group_size, batch_size):
    opt = []
    # f = open(file_path, 'rb')
    # data = pickle.load(f)
    for i in range(n_layers):
        qkv_bit = dot_bit = ffn1_bit = ffn2_bit = int(precision_bits[1]/2)
        # qkv dequant(element-wise product)
        
        qkv = [[1, hidden_states] , [hidden_states*3, hidden_states] , [1, hidden_states*3] , [] , [],  qkv_bit , 1]
        qkv_dequant = [[1, int(hidden_states/batch_size)] , [int(hidden_states*3/group_size), int(hidden_states/batch_size)] , [1, int(hidden_states*3/group_size)] , [] , [],  precision_bits[1] , 1]

        # output projection dequant(element-wise product)
        
        dot_product = [[1, hidden_states] , [hidden_states, hidden_states] , [1, hidden_states] , [] , [],  dot_bit , 1]
        dot_product_dequant = [[1, int(hidden_states/batch_size)] , [int(hidden_states/group_size), int(hidden_states/batch_size)] , [1, int(hidden_states/group_size)] , [] , [],  precision_bits[1] , 1]
        
        # ffn1 dequant
        
        ffn_1 = [[1, hidden_states] , [hidden_states*4, hidden_states] , [1, hidden_states*4] , [] , [],  ffn1_bit , 1] 
        ffn_1_dequant = [[1, int(hidden_states/batch_size)] , [int(hidden_states*4/group_size), int(hidden_states/batch_size)] , [1, int(hidden_states*4/group_size)] , [] , [], precision_bits[1] , 1]
        
        # ffn2 dequant
        
        ffn_2 = [[1, hidden_states*4] , [hidden_states, hidden_states*4] , [1, hidden_states] , [] , [],  ffn2_bit , 1]
        ffn_2_dequant = [[1, int(hidden_states*4/batch_size)] , [int(hidden_states/group_size), int(hidden_states*4/batch_size)] , [1, int(hidden_states/group_size)] , [] , [],  precision_bits[1] , 1]
        
        opt.extend((qkv, qkv_dequant, dot_product, dot_product_dequant, ffn_1, ffn_1_dequant, ffn_2, ffn_2_dequant))

    return opt

def gen_bitfusion(n_layers, hidden_states, batch_size):
    opt = []
    # f = open(file_path, 'rb')
    # data = pickle.load(f)
    index = random.sample(list(range(n_layers)), int(n_layers/2))
    for i in range(n_layers):
        if i in index:
            dot_bit = 4
        else:
            dot_bit = 8
        qkv_bit = ffn1_bit = ffn2_bit = 8
        qkv_dequant_bit = dot_dequant_bit = ffn1_dequant_bit = ffn2_dequant_bit = 8
        # qkv dequant(element-wise product)
        qkv_dequant = [[1, hidden_states] , [int(hidden_states*3/batch_size), hidden_states] , [1, int(hidden_states*3/batch_size)] , [] , [],  qkv_dequant_bit , 1]
        qkv = [[1, hidden_states] , [hidden_states*3, hidden_states] , [1, hidden_states*3] , [] , [],  qkv_bit , 1]
        
        # output projection dequant(element-wise product)
        dot_product_dequant = [[1, hidden_states] , [int(hidden_states/batch_size), hidden_states] , [1, int(hidden_states/batch_size)] , [] , [],  dot_dequant_bit , 1]
        dot_product = [[1, hidden_states] , [hidden_states, hidden_states] , [1, hidden_states] , [] , [],  dot_bit , 1]
        
        # ffn1 dequant
        ffn_1_dequant = [[1, hidden_states] , [int(hidden_states*4/batch_size), hidden_states] , [1, int(hidden_states*4/batch_size)] , [] , [],  ffn1_dequant_bit , 1]
        ffn_1 = [[1, hidden_states] , [hidden_states*4, hidden_states] , [1, hidden_states*4] , [] , [],  ffn1_bit , 1]
        
        # ffn2 dequant
        ffn_2_dequant = [[1, hidden_states*4] , [int(hidden_states/batch_size), hidden_states*4] , [1, int(hidden_states/batch_size)] , [] , [],  ffn2_dequant_bit , 1]
        ffn_2 = [[1, hidden_states*4] , [hidden_states, hidden_states*4] , [1, hidden_states] , [] , [],  ffn2_bit , 1]
        
        opt.extend((qkv_dequant, qkv, dot_product_dequant, dot_product, ffn_1_dequant, ffn_1, ffn_2_dequant, ffn_2))

    return opt

def gen_ant(n_layers, hidden_states, batch_size):
    opt = []
    # f = open(file_path, 'rb')
    # data = pickle.load(f)
    index = random.sample(list(range(n_layers)), int(n_layers/5))
    for i in range(n_layers):
        if i not in index:
            dot_bit = 4
        else:
            dot_bit = 8
        qkv_bit = ffn1_bit = ffn2_bit = 8
        qkv_dequant_bit = dot_dequant_bit = ffn1_dequant_bit = ffn2_dequant_bit = 8
        # qkv dequant(element-wise product)
        qkv_dequant = [[1, hidden_states] , [int(hidden_states*3/batch_size), hidden_states] , [1, int(hidden_states*3/batch_size)] , [] , [],  qkv_dequant_bit , 1]
        qkv = [[1, hidden_states] , [hidden_states*3, hidden_states] , [1, hidden_states*3] , [] , [],  dot_bit , 1]
        
        # output projection dequant(element-wise product)
        dot_product_dequant = [[1, hidden_states] , [int(hidden_states/batch_size), hidden_states] , [1, int(hidden_states/batch_size)] , [] , [],  dot_dequant_bit , 1]
        dot_product = [[1, hidden_states] , [hidden_states, hidden_states] , [1, hidden_states] , [] , [],  dot_bit , 1]
        
        # ffn1 dequant
        ffn_1_dequant = [[1, hidden_states] , [int(hidden_states*4/batch_size), hidden_states] , [1, int(hidden_states*4/batch_size)] , [] , [],  ffn1_dequant_bit , 1]
        ffn_1 = [[1, hidden_states] , [hidden_states*4, hidden_states] , [1, hidden_states*4] , [] , [],  dot_bit , 1]
        
        # ffn2 dequant
        ffn_2_dequant = [[1, hidden_states*4] , [int(hidden_states/batch_size), hidden_states*4] , [1, int(hidden_states/batch_size)] , [] , [],  ffn2_dequant_bit , 1]
        ffn_2 = [[1, hidden_states*4] , [hidden_states, hidden_states*4] , [1, hidden_states] , [] , [],  dot_bit , 1]
        
        opt.extend((qkv_dequant, qkv, dot_product_dequant, dot_product, ffn_1_dequant, ffn_1, ffn_2_dequant, ffn_2))

    return opt

def gen_benchmark(mix, precision_bits, batch_size, model_name = None):
    if mix:
        vit = gen_opt_mix(12, 768, precision_bits, 128, batch_size)
        vit = [[[1, 3, 224, 224] , [768, 3, 16, 16] , [1, 768, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit
        
        vit_huge = gen_opt_mix(32, 1280, precision_bits, 128, batch_size)
        vit_huge = [[[1, 3, 224, 224] , [1280, 3, 16, 16] , [1, 1280, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit_huge
        
        opt = gen_opt_mix(32, 4096, precision_bits, 128, batch_size)

        llama = gen_opt_mix(40, 5120, precision_bits, 128, batch_size)
    else:
        if not model_name:
            vit = gen_opt(12, 768, precision_bits, batch_size)
            vit = [[[1, 3, 224, 224] , [768, 3, 16, 16] , [1, 768, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit
        
            vit_huge = gen_opt(32, 1280, precision_bits, batch_size)
            vit_huge = [[[1, 3, 224, 224] , [1280, 3, 16, 16] , [1, 1280, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit_huge
    
            opt = gen_opt(32, 4096, precision_bits, batch_size)

            llama = gen_opt(40, 5120, precision_bits, batch_size)
        elif model_name == 'Bitfusion':
            vit = gen_bitfusion(12, 768, batch_size)
            vit = [[[1, 3, 224, 224] , [768, 3, 16, 16] , [1, 768, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit
        
            vit_huge = gen_bitfusion(32, 1280, batch_size)
            vit_huge = [[[1, 3, 224, 224] , [1280, 3, 16, 16] , [1, 1280, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit_huge
    
            opt = gen_bitfusion(32, 4096, batch_size)

            llama = gen_bitfusion(40, 5120, batch_size)
        elif model_name == 'ANT':
            vit = gen_ant(12, 768, batch_size)
            vit = [[[1, 3, 224, 224] , [768, 3, 16, 16] , [1, 768, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit
        
            vit_huge = gen_ant(32, 1280, batch_size)
            vit_huge = [[[1, 3, 224, 224] , [1280, 3, 16, 16] , [1, 1280, 14, 14] , [16, 16] , [0, 0] ,  precision_bits[1] , 0]] + vit_huge
    
            opt = gen_ant(32, 4096, batch_size)

            llama = gen_ant(40, 5120, batch_size)
    
    print("vit = [")
    for i in vit:
        print(str(i)+",")
    print("]")
    print("\n")
    
    print("vit_huge = [")
    for i in vit_huge:
        print(str(i)+",")
    print("]")
    
    print("opt = [")
    for i in opt:
        print(str(i)+",")
    print("]") 
    
    print("llama = [")
    for i in llama:
        print(str(i)+",")
    print("]")

def calculate_dequant(precision_bits, batch_size):
    opt_mix = gen_opt_mix(32, 4096, precision_bits, 128, batch_size)
    opt = gen_opt(32, 4096, precision_bits, batch_size)
    dequant_mix = dequant = 0
    compute = compute_mix = 0
    if precision_bits[1] == 8:
        compute_bits = 4
    else:
        compute_bits = 6
    for i in range(len(opt_mix)):
        if i%2 == 0:
            dequant_mix += (opt_mix[i][1][0] * opt_mix[i][1][1])*precision_bits[1]
            compute += opt[i][1][0] * opt[i][1][1]*precision_bits[1]
        else:
            compute_mix += opt_mix[i][1][0] * opt_mix[i][1][1]*compute_bits
            dequant += opt[i][1][0] * opt[i][1][1]*precision_bits[1]
    dequant_mix_ratio = dequant_mix/(dequant_mix+compute_mix)
    compute_mix_ratio = compute_mix/(dequant_mix+compute_mix)
    dequant_ratio = dequant/(dequant+compute)
    compute_ratio = compute/(dequant+compute)
    return dequant_mix_ratio, compute_mix_ratio, dequant_ratio, compute_ratio
# #int88
# gen_benchmark(False, [4,8], 8)

# #int48
# gen_benchmark(True, [4,8], 8)

# #mix float
# gen_benchmark(True, [4,16], 8)

# float
# gen_benchmark(False, [16,16], 8)

# # Olaccel
# gen_benchmark(False, [4,8], 8)

# BitFusion
# gen_benchmark(False, [4,8], 8, True)

gen_benchmark(False, [4,8], 8, 'ANT')
# print(calculate_dequant([4,8], 1))
# print(calculate_dequant([4,8], 2))
# print(calculate_dequant([4,8], 4))
# print(calculate_dequant([4,8], 6))
# print(calculate_dequant([4,8], 8))
# print(calculate_dequant([4,8], 16))
# print(calculate_dequant([4,8], 32))

# print(calculate_dequant([4,16], 2))
# print(calculate_dequant([4,16], 4))
# print(calculate_dequant([4,16], 8))
# print(calculate_dequant([4,16], 16))
# print(calculate_dequant([4,16], 32))