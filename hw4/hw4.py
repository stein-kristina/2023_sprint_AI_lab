"""
第4次作业, 选择其中一种算法实现即可.
Puzzle问题的输入数据类型为二维嵌套list, 空位置用 `0`表示. 输出的解数据类型为 `list`, 是移动数字方块的次序.
"""
import copy as cp
import timeit as ti
import heapq


class Node:
    def __init__(self, g_n, h_n, board, road):
        self.g_n = g_n
        self.h_n = h_n
        self.board = board  # 当前棋局
        self.road = road  # 一直以来的路径
        self.f_n = self.g_n + (self.h_n * 2)

    def __lt__(self, other):
        if self.f_n == other.f_n:
            return self.g_n > other.g_n
        return self.f_n < other.f_n




def evaluate_h_n(puzzle, n: int) -> int:
    # 计算h_n，曼哈顿距离
    h = 0
    final = 0
    for i in range(n):
        for j in range(n):
            final += 1
            if puzzle[i][j] == 0 or puzzle[i][j] == final:
                continue
            else:
                x = int(int(puzzle[i][j]-1)/n)
                y = puzzle[i][j] - 1 - n*x
                # 你想去的位置刚好是我的
                px = int((puzzle[x][y]-1)/n)
                py = puzzle[x][y] -1 - n*px
                if px == i and py == j:
                    h += 2
                h += (abs(x-i) + abs(y-j))

    return h


def constru_tar(n):
    # 构建目标节点
    tar = []
    for i in range(n):
        A = []
        for j in range(n):
            A.append(i * n + j+1)
        tar.append(A)
    tar[n-1][n-1] = 0
    return tar


def find_zero(board, n):
    # 找0
    for i in range(n):
        if 0 in board[i]:
            return i, board[i].index(0)
    return -1, -1


def swap(board: list[list[int]], fx, fy, tx, ty):
    t = board[tx][ty]
    board[tx][ty] = board[fx][fy]
    board[fx][fy] = t
    return


many_node = 0
dxdy = [[-1, 0], [1, 0], [0, -1], [0, 1]]


def A_star(puzzle):
    global many_node, dxdy
    n = len(puzzle)
    start = Node(0, evaluate_h_n(puzzle, n), cp.deepcopy(puzzle), [])
    open = [start]  # 优先队列
    closed = set()  # 已经查找过

    heapq.heapify(open)  # 最小堆
    end = str(constru_tar(n))  # 最终状态
    quick_open = {str(puzzle): 0}  # 棋盘:g_n

    while len(open):
        head = heapq.heappop(open)
        s = str(head.board)
        if s in closed:
            continue
        quick_open.pop(s)
        closed.add(s)
        if s == end:
            return head.road
        zero_i, zero_j = find_zero(head.board, n)
        # 上，下，左，右都可以移进去
        for i in range(4):
            ix = zero_i + dxdy[i][0]
            iy = zero_j + dxdy[i][1]
            if 0 <= ix < n and 0 <= iy < n:
                swap(head.board, ix, iy, zero_i, zero_j)
                ss = str(head.board)
                if ss in closed:
                    pass
                elif ss in quick_open and quick_open[ss] > head.g_n + 1:
                    # 更新open内部的g(s')
                    heapq.heappush(open, Node(head.g_n+1, head.h_n, head.board, head.road))
                    quick_open[ss] = head.g_n + 1
                else:
                    new_road = cp.deepcopy(head.road)
                    new_road.append(head.board[zero_i][zero_j])
                    no_name = Node(head.g_n + 1, evaluate_h_n(head.board, n), cp.deepcopy(head.board), new_road)
                    many_node += 1  # 测试产生节点个数
                    heapq.heappush(open, no_name)
                    quick_open[ss] = head.g_n + 1

                swap(head.board, zero_i, zero_j, ix, iy)

    pass


many_node_2 = 0
def IDA_star(puzzle):
    global many_node_2
    n = len(puzzle)
    closed = set()  # 已经查找过
    end = constru_tar(n)  # 最终状态
    dxdy = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    st = [Node(0, evaluate_h_n(puzzle, n), cp.deepcopy(puzzle), [])]  # 栈
    d_1 = 1  # 初始深度限制
    d_2 = 8888888888
    while 1:
        if d_2 < 8888888888:
            d_1 = d_2
            d_2 = 8888888888
            st = [Node(0, evaluate_h_n(puzzle, n), cp.deepcopy(puzzle), [])]
        while len(st):
            head = st[-1]
            st.pop()
            closed.add(str(head.board))
            if head.board == end:
                return head.road
            else:
                zero_i, zero_j = find_zero(head.board, n)
                for i in range(4):
                    ix = zero_i + dxdy[i][0]
                    iy = zero_j + dxdy[i][1]
                    if 0 <= ix < n and 0 <= iy < n:
                        swap(head.board, ix, iy, zero_i, zero_j)
                        if str(head.board) in closed:
                            pass
                        else:
                            new_road = cp.deepcopy(head.road)
                            new_road.append(head.board[zero_i][zero_j])
                            eva = evaluate_h_n(head.board, n) + head.g_n + 1  # f_n
                            if eva <= d_1:
                                st.append(Node(head.g_n + 1, evaluate_h_n(head.board, n), cp.deepcopy(head.board), new_road))
                                many_node_2 += 1
                            else:
                                d_2 = min(d_2, eva)
                        swap(head.board, zero_i, zero_j, ix, iy)
        closed.clear()
        if len(st) == 0 and d_2 == 8888888888:
            return [-1]  # 搜索失败
    pass


if __name__ == '__main__':
    import time
    print("go task0")
    puzzle = [[1,2,3,4],[5, 6, 7, 8], [9, 10, 11, 12],[0, 13, 14, 15]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task1")  # 理论作业
    puzzle = [[5, 1, 2, 4], [9, 6, 3, 8], [13, 15, 10, 11], [14, 0, 7, 12]]
    start = time.time()
    sol = A_star(puzzle)
    end = time.time()
    print("time:", end-start)
    print(len(sol), sol)

    print("go task2")
    puzzle = [[8,10,12,13],[4,5,9,14],[6,0,11,1],[3,2,15,7]]
    start=time.time()
    many_node = 0
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol, many_node)

    print("go task3")
    puzzle = [[14, 10, 6, 0],[4, 9 ,1 ,8],[2, 3, 5 ,11],[12, 13, 7 ,15]]
    start = time.time()
    sol = A_star(puzzle)
    end = time.time()
    print("time:", end-start)
    print(len(sol), sol)

    print("go task4")
    puzzle = [[6, 10, 3, 15],[14, 8, 7, 11], [5, 1, 0, 2],[13, 12, 9, 4]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task5")
    puzzle = [[11, 3, 1, 7],[4, 6, 8, 2], [15, 9, 10, 13],[14, 12, 5, 0]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task6")
    puzzle = [[0, 5, 15, 14],[7, 9, 6, 13], [1, 2, 12, 10],[8, 11, 4, 3]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)


