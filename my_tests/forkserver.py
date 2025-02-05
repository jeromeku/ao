# import multiprocessing

# global_var = "Original"
# print(f"Top-level global_var: {global_var}")  # This will print in both parent and child

# def worker():
#     print(f"Child sees: {global_var}")  # Will NOT be "Changed"

# if __name__ == "__main__":
#     print("Parent process running")
    
#     ctx = multiprocessing.get_context("spawn")
    
#     global_var = "Changed"  # Modify only in parent

#     p = ctx.Process(target=worker)
#     p.start()
#     p.join()

import multiprocessing

global_var = "Original"

def worker():
    print(f"Child sees: {global_var}")  # Expected to see "Original", not "Changed"
    print(f"Child sees: {main_var}")  # Expected to see "Hello", not "Changed"
if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")

    # Change global variable in parent
    global_var = "Changed"
    main_var = "Hello"

    p = ctx.Process(target=worker)
    p.start()
    p.join()