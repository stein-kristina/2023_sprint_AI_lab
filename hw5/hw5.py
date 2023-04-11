"""
第5次作业, 用遗传算法解决TSP问题
本次作业可使用`numpy`库和`matplotlib`库以及python标准库
请不要修改类名和方法名
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import time as ti
import copy as cp
import math


class GeneticAlgTSP:
    def __init__(self, tsp_filename):
        self.cities = [[0, 0]]  # 读取文件数据, 存储到该成员中
        # 下标映射，存储的元素为元组，即二维坐标

        self.population = []  # 初始化种群, 会随着算法迭代而改变
        self.cities_num = None
        self.popsize = None  # 种群数目
        with open(tsp_filename) as file_object:
            f = 0  # 判定到没到数据段
            for line in file_object.readlines():
                if not f:
                    if line.find('DIMENSION') != -1:
                        index = len('DIMENSION')
                        # 找数字
                        while line[index] < '0' or line[index] > '9':
                            index += 1
                        line = line.rstrip('\n')
                        self.cities_num = int(line[index:])
                    line = line.rstrip('\n')
                    if line == 'NODE_COORD_SECTION':
                        f = 1  # 进入读取模式
                else:
                    line = line.rstrip('\n')
                    if line == 'EOF':
                        break
                    dxdy = list(line.split(' '))
                    dxdy[1] = float(dxdy[1])
                    dxdy[2] = float(dxdy[2])
                    self.cities.append(dxdy[1:])
        self.cities = np.array(self.cities)

        self.distance = np.array([0.0 for i in range((2 + self.cities_num) * (self.cities_num + 1) // 2)])  # 距离三角矩阵
        for i in range(2, self.cities_num + 1):
            for j in range(1, i):
                self.distance[self.new_index(i, j)] = self.two_city_dis(self.cities[i], self.cities[j])  # 距离矩阵

        self.popsize = 100
        # 贪心初始化
        self.greedy_init(self.popsize)

    def greedy_init(self, init_num):  # 贪婪生成解
        solutions = []
        distance = []
        i = 0
        while i < init_num:
            cur = random.randint(1, self.cities_num)
            pop = [cur]  # 路径
            rest = set([x for x in range(1, self.cities_num + 1)])
            rest.remove(cur)
            i_cur = cur
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if self.distance[self.new_index(cur, x)] < tmp_min:
                        tmp_min = self.distance[self.new_index(cur, x)]
                        tmp_choose = x
                # for x in rest:
                #     z = self.two_city_dis(self.cities[cur], self.cities[x])
                #     if z < tmp_min:
                #         tmp_min = z
                #         tmp_choose = x
                cur = tmp_choose
                pop.append(tmp_choose)
                rest.remove(tmp_choose)
            pop.append(i_cur)
            fpop = self.get_fitness(pop)
            solutions.append(pop)
            distance.append(fpop)
            i += 1
        self.population = (np.array(solutions), np.array(distance))
        return

    def new_index(self, i, j) -> int:
        #  i > j
        if i < j:
            t = i
            i = j
            j = t
        return int((1 + i) * i / 2 + j)

    def generate_path(self, n):
        path = list(range(1, n + 1))
        random.shuffle(path)
        return path

    def two_city_dis(self, city_1: np.array, city_2: np.array) -> float:  # 返回两个城市的距离
        return float(float((city_1[0] - city_2[0]) ** 2 + (city_1[1] - city_2[1]) ** 2) ** 0.5)

    def path_dis(self, path) -> float:  # 计算路径长度
        ret = 0.0
        for i in range(self.cities_num):
            ret += self.distance[self.new_index(path[i], path[i + 1])]
            # ret += self.two_city_dis(self.cities[path[i]], self.cities[path[i+1]])
        return ret

    def get_fitness(self, pop):
        return self.path_dis(pop)  # 直接就是正的

    def Inversion_Mutation(self, child, mutation_rate):  # 有可能变异，浅复制
        s = random.randint(1, self.cities_num - 2)
        t = random.randint(s + 1, self.cities_num - 1)
        if random.random() < mutation_rate:  # 开始变异
            for i in range(s, int((s + t) / 2) + 1):
                st = child[i]
                child[i] = child[t + s - i]
                child[t + s - i] = st
        return child

    def Insert_mutation(self, child: list, mutation_rate):
        if random.random() < mutation_rate:  # 开始变异
            s = random.randint(1, self.cities_num - 2)
            t = random.randint(s + 1, self.cities_num - 1)
            temp = child[t]
            for i in range(t - 1, s - 1, -1):
                child[i + 1] = child[i]  # 将r处插入到l+1处,并逐个移动
            child[s] = temp
        return child

    def Two_reverse(self, old_child, i, j):  # 2-opt优化,反转一段基因序列
        new_child = cp.deepcopy(old_child)
        l = i + 1
        r = j
        while l < r:
            temp = new_child[l]
            new_child[l] = new_child[r]
            new_child[r] = temp
            l += 1
            r -= 1
        return new_child

    def Two_optimize(self, child, mutation_rate, max_iter):
        if random.random() < mutation_rate:
            cost = self.get_fitness(child)  # 距离
            for t in range(max_iter):  # 最大迭代次数
                min_change = 0
                min_ij = None
                for i in range(0, self.cities_num - 2):  # 找到最优的交换
                    for j in range(i+2, self.cities_num):
                        if i == 0 and j == self.cities_num-1:
                            continue
                        # change = self.two_city_dis(self.cities[child[i]], self.cities[child[j]]) + \
                        #          self.two_city_dis(self.cities[child[i + 1]], self.cities[child[j + 1]])\
                        #          - self.two_city_dis(self.cities[child[i]], self.cities[child[i + 1]])\
                        #          - self.two_city_dis(self.cities[child[j]], self.cities[child[j + 1]])
                        change = self.distance[self.new_index(child[i], child[j])] + self.distance[self.new_index(child[i+1], child[j+1])] - self.distance[self.new_index(child[i], child[i + 1])] - self.distance[self.new_index(child[j], child[j+1])]
                        if change < min_change:
                            min_change = change
                            min_ij = (i, j)
                if min_change == 0:  # 没有优化了
                    break
                new_child = self.Two_reverse(child, min_ij[0], min_ij[1])
                cost += min_change
                child = new_child
            return child

    def Partial_Mapped_crossover(self, parent_1: np.array, parent_2: np.array):  # 部分映射交叉，两点
        # 随机生成两个下标s，t
        parent_1 = parent_1.tolist()  # 转换为链表操作
        parent_2 = parent_2.tolist()  # 转换为链表操作
        s = random.randint(0, self.cities_num - 2)
        t = random.randint(s + 1, self.cities_num - 1)
        map_p1 = dict()  # 确定映射关系
        parent_1.pop()  # 把头城市去掉先
        parent_2.pop()
        # s到t段替换
        for i in range(s, t + 1):
            st = parent_1[i]
            parent_1[i] = parent_2[i]
            parent_2[i] = st
            map_p1[parent_1[i]] = parent_2[i]
        # 前s段替换
        for i in range(0, s):
            p = parent_1[i]
            head = p  # 起始位置
            while map_p1.get(p, -1) != -1:
                if map_p1[p] == head:  # 又回到了起点
                    p = head
                    break
                p = map_p1[p]
            parent_1[i] = p
        # 后t段替换
        for j in range(t + 1, self.cities_num):
            p = parent_1[j]
            head = p  # 起始位置
            while map_p1.get(p, -1) != -1:
                if map_p1[p] == head:  # 又回到了起点
                    p = head
                    break
                p = map_p1[p]
            parent_1[j] = p
        parent_1.append(parent_1[0])
        return parent_1  # 生成儿子

    def Order_crossover(self, parent_1: np.array, parent_2: np.array):
        # OX3
        parent_1 = parent_1.tolist()  # 转换为链表操作
        parent_2 = parent_2.tolist()  # 转换为链表操作
        s = random.randint(0, self.cities_num - 2)
        t = random.randint(s + 1, self.cities_num - 1)
        length = t - s + 1  # 长度
        s2 = random.randint(0, self.cities_num - length)
        parent_1.pop()
        parent_2.pop()
        delete_s1 = set()
        for i in range(s, t + 1):
            delete_s1.add(parent_1[i])

        child_1 = [0] * (self.cities_num)
        p1 = 0  # child_1用于在parent_2中寻找交换的指针
        for i in range(0, self.cities_num):
            if s <= i <= t:
                child_1[i] = parent_1[i]
            else:
                while p1 < self.cities_num and parent_2[p1] in delete_s1:
                    p1 += 1
                if p1 < self.cities_num:
                    child_1[i] = parent_2[p1]
                    p1 += 1
        child_1.append(child_1[0])
        return child_1

    def tournament_select(self, elite_num, pop_size, tournament_size):
        new_pops = []
        # 步骤1 从群体中随机选择M个个体，计算每个个体的目标函数值
        search = [i for i in range(0, pop_size)]
        while len(new_pops) < elite_num:
            tournament_list = random.sample(search, tournament_size)
            tour = []
            for i in range(tournament_size):
                tour.append(self.population[tournament_list[i]])
            # 步骤2 根据每个个体的目标函数值，计算其适应度
            tour.sort(key=lambda x: x[self.cities_num], reverse=True)
            # 步骤3 选择适应度最大的个体
            pop = tour[0]
            new_pops.append(pop)
        return new_pops

    def tour_select(self, elite_num):
        temp = [i for i in range(0, elite_num)]
        selectit = []
        for i in range(2):
            test = np.random.choice(temp, 2)
            index = np.argmin(self.population[1][test])
            selectit.append(self.population[0][index])
            temp.remove(test[index])
        return selectit[0], selectit[1]

    def iterate(self, num_iterations):
        # 基于self.population进行迭代,返回当前较优路径
        elite_num = self.popsize  # 种群个数
        mutation_rate = 1  # 变异概率
        best_one = None
        best_fitness = 10**8

        for t in range(num_iterations):  # 迭代次数
            new_population = cp.deepcopy(self.population)

            for i in range(elite_num):
                p1, p2 = self.tour_select(elite_num)  # list
                child = None

                choose1 = random.random()
                if choose1 < 0.2:
                    child = self.Order_crossover(p1, p2)
                else:
                    child = self.Partial_Mapped_crossover(p1, p2)

                choose2 = random.random()
                if choose2 < 0.01:
                    max_iter = 200
                    child = self.Two_optimize(child, mutation_rate, max_iter)
                elif 0.01 <= choose2 < 0.1:
                    child = self.Inversion_Mutation(child, mutation_rate)
                elif 0.1 <= choose2 < 0.25:
                    child = self.Insert_mutation(child, mutation_rate)
                cost = self.get_fitness(child)

                if np.isin([cost], new_population[1])[0]:
                    continue

                new_population[0][i] = child
                new_population[1][i] = cost

            temp1 = np.argsort(self.population[1])
            temp2 = np.argsort(-new_population[1])
            for i in range(0, int(0.05*elite_num)):
                new_population[0][temp2[i]] = self.population[0][temp1[i]]
                new_population[1][temp2[i]] = self.population[1][temp1[i]]

            self.population = new_population  # 更换种群

            best_index = np.argmin(self.population[1])
            if best_fitness > self.population[1][best_index]:
                best_fitness = self.population[1][best_index]
                best_one = self.population[0][best_index]

        return best_one.tolist()[0:-1]

    def draw_pic(self, T, best_one: list[int]):
        x = []
        y = []
        # best_one = best_one.tolist()
        best_one.append(best_one[0])
        for i in range(self.cities_num+1):
            x.append(self.cities[best_one[i]][0])
            y.append(self.cities[best_one[i]][1])
        plt.plot(x, y, color='g', marker='s', linestyle='', label='city', markersize=5)
        plt.legend()

        plt.plot(x, y, color='b', linestyle='-', linewidth=2)
        plt.xlabel('city_x', loc='center')
        plt.ylabel('city_y', loc='center')
        road = self.path_dis(best_one)
        plt.title('iteration:' + str(T) + '\n' + 'distance:' + str(road))
        plt.show()
        pass


if __name__ == "__main__":
    T = 1
    total_T = 10
    use_time = [0 for i in range(3)]  # 时钟
    pic_x = []
    pic_y = []
    result = []
    up = 1
    best = None
    for k in range(up):
        apex = None
        start = ti.time()
        tsp = GeneticAlgTSP(
            "C:\\Users\\86189\\Desktop\\大二下\\人工智能\\2023AI\\实验课\\Homework\\hw5\\uy734.tsp")  # 读取Djibouti城市坐标数据
        end = ti.time()
        print(end - start)
        for i in range(total_T):
            start = ti.time()
            tour = tsp.iterate(T)  # 对算法迭代10次
            end = ti.time()
            use_time[2] += float(end - start)
            add_2 = int(use_time[2] / 60) % 60
            use_time[1] += add_2
            add_1 = int(use_time[1] / 60)
            use_time[0] += add_1
            use_time[1] = use_time[1] % 60
            use_time[2] -= add_1 * 3600 + add_2 * 60
            best = cp.deepcopy(tour)
            tour.append(tour[0])
            dis = tsp.path_dis(tour)

            pic_x.append((i + 1) * T)
            pic_y.append(dis)
            apex = dis


            print((i + 1) * T, ' ', 'time:', use_time[0], 'h', use_time[1], 'm', use_time[2], 's', ' dis:',
                  dis)
        result.append(apex)
        tsp.draw_pic(T * total_T, best)
    # 迭代曲线
    plt.plot(pic_x, pic_y, color='r', linestyle='-', label='city')
    plt.title('qa194.tsp')
    plt.show()
    # plt.plot(range(0, up), result, color='r', linestyle='-')
    # plt.show()
