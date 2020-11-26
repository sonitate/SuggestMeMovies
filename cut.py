from csv import writer
from csv import reader
import pandas as pd
import json 
def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)

# add_column_in_csv('meta_and_keywords_clean.csv', 'output_3.csv', lambda row, line_num: row.append(row[0] + '__' + row[1]))

# header_of_new_col = 'genres_cut'
# default_text = 'Some_Text'
# # Add the column in csv file with header
# add_column_in_csv('input_with_header.csv', 'output_6.csv',
#                   lambda row, line_num: row.append(header_of_new_col) if line_num == 1 else row.append(
#                       default_text))

column_names = ['ID','Title','Date','Language','Genre','averageRating','numVotes','Keywords']
df = pd.read_csv("meta_and_keywords_clean.csv", names=column_names)

letters = df.Genre.to_list()
# for i , j in letters[1].items():
#     print(i)
#     print(j)
letters_clean = []

for d in letters[1:]:
    res = json.loads(d)
    letters_clean.append(res)

# list_str_group = ['genres_clean']
print(letters_clean[10])
list_str_group = [' '.join(i.values()) for i in letters_clean]
j=0
i=0
list_index = []
for i in letters_clean:
    list_index.append(j)
    j+=1
print(list_index[3])
# res = sum(test_dict.values(), []) 
print(list_str_group[10])
hello = [x.strip(' ') for x in list_str_group]
header_of_new_col = 'genres_clean'
# default_text = 'Some_Text'
# # Add the column in csv file with header
add_column_in_csv('out1.csv', 'meta_and_keywords_clean_cut.csv',lambda row, line_num: row.append(header_of_new_col) if line_num == 1 else row.append(hello[line_num -2]))
# add_column_in_csv('meta_and_keywords_clean.csv', 'out1.csv',lambda row, line_num: row.append('index') if line_num == 1 else row.append(list_index[line_num -2]))
# add_column_in_csv('meta_and_keywords_clean.csv', 'output_4.csv', lambda row, line_num: row.append(list_use[line_num - 1]))sx