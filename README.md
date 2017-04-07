# experiments

## Docker

* __Link:__ https://hub.docker.com/r/nikitxskv/comparison/
* __How to run__:

```
docker run --workdir /root -v <path_to_local_folder>:/root/shared -p 80:8888 -it nikitxskv/comparison sh -c "ipython notebook --ip=* --no-browser"
```

## Files

* __experiment.py__:
	* Родительский класс, в котором реализованы функции считывания данных, разбиение на подвыборки, подсчет счетчиков, функция кросс-валидации и функция run, которая запускает эксперимент.

* __\*_experiment.py__:
	* Дочерние классы, в которых реализованы функции перевода данных под формат конкретного алгоритма, задание перебираемых параметров и распределений, откуда они выбираются, а так же функции запуска обучения.

* __run.py__:
	* Файл запуска экспериментов.
	* Как запускать:
	```
	python run.py <model> <learning_task> <path_to_dataset> <output_folder> <number_of_hyperopt_runs> <number_of_estimators>
	```
	* Пример:
	```
	python run.py xgb classification ./adult/ ./ 500 5000
	```

* __install.sh__:
	* Скрипт, который устанавливает все необходимое для проведения экспериментов, а так же XGBoost и LightGBM.

* __comparison_description.\*__:
	* PDF и TEX файлы с описанием формата сравнения.