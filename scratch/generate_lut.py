import torch


def generate_lookup_table(lookup_values):
    """
    Given a list of lookup values (as strings), generate a nested tl.where
    statement for a lookup table. The generated statement will be of the form:

      y = tl.where(x < 1, v1, tl.where(x < 2, v2, tl.where(x < 3, v3, ... vN)))

    where each value corresponds to the threshold condition:
      - If x < 1, then use lookup_values[0] (v1)
      - Else if x < 2, then use lookup_values[1] (v2)
      - ...
      - Else use the final value lookup_values[-1] (vN)
    
    Parameters:
      lookup_values (list of str): A list of lookup value strings.

    Returns:
      str: A string representing the nested tl.where statement.
    """
    if not lookup_values:
        raise ValueError("At least one lookup value must be provided.")

    # Start with the default value (the last value in the list).
    expr = lookup_values[-1]
    # Build the nested expression from the end toward the beginning.
    for i in range(len(lookup_values) - 1, 0, -1):
        # For each i, we use threshold i and value lookup_values[i-1].
        expr = f"tl.where(x < {i}, {lookup_values[i-1]}, {expr})"
    
    # Prepend 'y = ' to complete the assignment.
    return "y = " + expr

# Example usage:
lookup_values = torch.randn(10, dtype=torch.float32).tolist()
lookup_statement = generate_lookup_table(lookup_values)
print(lookup_statement)
