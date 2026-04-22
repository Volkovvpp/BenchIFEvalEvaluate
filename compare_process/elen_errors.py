import csv
import os


def extract_errors(input_filename, output_filename):
    target_column = 'prompt_level_strict_acc'

    if not os.path.exists(input_filename):
        print(f"Ошибка: Файл '{input_filename}' не найден в текущей папке.")
        return

    print(f"Читаю файл: {input_filename}...")

    error_count = 0

    try:
        with open(input_filename, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            error_rows = [row for row in reader if row[target_column].strip() == 'False']

            if not error_rows:
                print("Ошибок не найдено. Файл с результатами не будет создан.")
                return

            with open(output_filename, mode='w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(error_rows)

            print(f"Готово! Найдено ошибок: {len(error_rows)}")
            print(f"Строки с ошибками сохранены в файл: {output_filename}")

    except KeyError:
        print(f"Ошибка: В файле нет колонки с названием '{target_column}'.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")


file_to_read = 'compare_process/results-6970986b4e761c9b775f86c6.csv'
file_to_save = 'elen_errors.csv'

extract_errors(file_to_read, file_to_save)