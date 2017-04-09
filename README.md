# experiments

## Docker

* Докер со всем необходимым для запуска экспериментов.
* __Ссылка:__ https://hub.docker.com/r/nikitxskv/comparison/
* __Как запускать__:

```
docker run --workdir /root -v <path_to_local_folder>:/root/shared -p 80:8888 -it nikitxskv/comparison sh -c "ipython notebook --ip=* --no-browser"
```

## Files

* __experiment.py__:
    * Родительский класс, в котором реализованы функции считывания данных, разбиение на подвыборки, подсчет счетчиков, функция кросс-валидации и функция run, которая запускает эксперимент.

* __\*\_experiment.py__:
    * Дочерние классы, в которых реализованы функции перевода данных под формат конкретного алгоритма, задание перебираемых параметров и распределений, откуда они выбираются, а так же функции запуска обучения.

* __run.py__:
    * Файл запуска экспериментов.

* __install.sh__:
    * Скрипт, который устанавливает все необходимое для проведения экспериментов, а так же XGBoost и LightGBM.

* __comparison_description.\*__:
    * PDF и TEX файлы с описанием формата сравнения.

## Как запускать

* Можно запускать либо из командной строки, либо из интерпретатора, импортируя нужный класс.

* Параметры:
    ```
    Позиционные (обязательные) аргументы:
      algo                           Имя алгоритма {xgb, lgb}
      learning_task                  Вид задачи {classification, regression}

    Опциональные аргументы:
      -h [ --help ]                  Хелп
      -i [ --dataset_path ]          Путь к папке с датасетом
      -o [ --output_folder_path ]    Путь к папке с результатом
      -t [ --n_estimators ]          Количество деревьев (int, по умолчанию 2000)
      -n [ --n_iters ]               Количество итераций hyperopt'a (int, по умолчанию 50)
      -s [ --save_pred ]             Сохранять предсказания на тесте (bool, по умолчанию False)
      --holdout                      Размер Holdout части (float, по умолчанию -1 (не используется))
    ```

* Usage:
    ```
    python run.py algo learning_task [-h] [-i DATASET_PATH] [-o OUTPUT_FOLDER_PATH] [-t N_ESTIMATORS]
              [-n N_ITERS] [--holdout HOLDOUT] [-s]
    ```

* Примеры запуска
    * Из командной строки:
        ```
        python run.py xgb classification -i ./amazon/ -n 2 -t 10
        ```

    * Из интерпретатора:
        ```
        from xgboost_experiment import XGBExperiment
        xgb_exp = XGBExperiment("classification", "./amazon/", n_iters=2, n_estimators=10)
        xgb_exp.run()
        ```

