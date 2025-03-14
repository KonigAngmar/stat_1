from Ex_1 import task1
from Ex_2 import task2
from task3a import empty_boxes_test_a
from task3b import empty_boxes_test_b
from task4a import smirnov_test_a
from task4b import smirnov_test_b

if __name__ == "__main__":
    task1()
    task2()
    for n in [1000, 10000, 100000]:
        chi_stat_a, p_value_a = empty_boxes_test_a(n)
        print(f"Task 3a | n={n}: Chi-Square={chi_stat_a:.4f}, p-value={p_value_a:.4f}")

        chi_stat_b, p_value_b = empty_boxes_test_b(n)
        print(f"Task 3b | n={n}: Chi-Square={chi_stat_b:.4f}, p-value={p_value_b:.4f}")

        stat_a, p_value_a = smirnov_test_a(n)
        print(f"Task 4a | n={n}: Smirnov Stat={stat_a:.4f}, p-value={p_value_a:.4f}")

        stat_b, p_value_b = smirnov_test_b(n)
        print(f"Task 4b | n={n}: Smirnov Stat={stat_b:.4f}, p-value={p_value_b:.4f}")
