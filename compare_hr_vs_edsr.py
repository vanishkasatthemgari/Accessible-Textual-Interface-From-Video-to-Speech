def display_table():
    # Define table headers and data
    headers = ["Image Version", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    data = [
        ["High Resolution (Original)", 95.5, 95.83, 95.83, 95.83],
        ["EDSR Enhanced", 96.4, 100.0, 100.0, 100.0]
    ]
    
    # Calculate column widths
    column_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + data))]
    
    # Print the header
    header_row = " | ".join(f"{headers[i]:<{column_widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))  # Print separator line
    
    # Print each row of the data
    for row in data:
        print(" | ".join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

if __name__ == "__main__":
    display_table()
