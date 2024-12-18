Генерация изображений ладоней и рук.
Выполнил: Епифанов Антон Константинович Б05-408.

    Описание проекта.
Этот проект представляет собой веб-приложение для генерации изображений ладоней. Пользователи могут создавать уникальные изображения ладоней. Приложение использует современные технологии машинного обучения для генерации новых изображений рук и ладоней.

    Возможности.
- Генерация разного количества изображений ладоней и рук.
- Предпросмотр созданных изображений перед их сохранением.
- Возможность скачивания сгенерированных изображений в формате PNG.
- Интуитивно понятный пользовательский интерфейс.
- Возможность оценить качество генерации как отдельно взятой картинки, так и всего набора картинок.

    Технологии.
- Python 3.11, 3.12.
- HTML5
- CSS3
- Javascript
- Flask
- Pytorch

    Установка и запуск.
для установки сначала клонируйте репозиторий к себе на компьютер с помощью: git clone ssh_this_repository genPalmsWeb
Затем создайте новое виртуальное окружение и активируйте его:
    python -m venv venv
    source venv/bin/activate
Затем установите необходимые библиотеки:
    pip install -r requirements.txt
Теперь скачайте все нужные данные для обучения и работы сайта и обработайте их. Если вы не хотите тренировать модель, можете не запускать файл prepare_dataset_with_yolo.py:
    python src/training_model/load_all_data.py
    python src/training_model/prepare_dataset_with_yolo.py
Для запуска обучения напишите:
    python src/training_model/train_dcgan.py
Обучение займет достаточно много времени (от нескольких часов на мощных видеокартах до несколько дней или недель на более слабом оборудовании)
Для запуска самого сервиса надо написать и перейти по адресу localhost:5000:
    python src/server.py
Можно пользоваться.

  Архитектура.
Для улучшения обучения я обрабатываю исходные датасет, обрезая на нем края на изображениях с помощью YOLOv11, которую тоже обучаю на датасете с кэггла. Это нужно для того, чтобы сеть фокусировалась не на фоне, а на генерации самих линий на ладонях.
Модели, используемые для генерации изображений обучаются состязательно: сеть-дискриминатор учится отличать настоящие изображения ладоней и рук от сгенерированных, в то время как сеть-генератор старается научиться создавать такие изображения, на которых бы ошибался дискриминатор и принимал их за настоящие. Таким образом генератор старается увеличить лосс дискриминатора, а дискриминатор уменьшить свой лосс и при этом увеличить лосс генератора, так как лосс генератора есть не что иное как "обратный" лосс дискриминатора.
Модели реализованы в виде классов, которые наследуются от torch.nn.Module. Классы в качестве полей имеют свои слои, которыми модель обрабатывает данные. Само предсказание происходит в методе forward, который мы перегружаем. Само обучение происходит в методе train класса Trainer. Этот класс в качестве своих полей содержит все нужные данные для обучения: модели, которые надо обучить, данные на которых надо обучить и некоторые параметры обучения.
Сам веб-сервис построен довольно просто. Реализованы некоторые эндпоинты и по переходу на некоторый эндпоинт выполняются соответствующие действия: рендеринг нужного html шаблона и выполнение логики, зависящее от эндпоинта. Стартовая страница предлагает пользователю выбрать количество картинок, которое он хочет сгенерировать. Если пользователь вводит невалидные данные (не число или слишком большое число), его перекидывает на специальную страницу с ошибкой. Если все ок, то вызывается эндпоинт с генерацией и начинается генерация. После ее окончания пользователя перекидывает на страничку, где можно увидеть сгенерированные изображения, а также вероятность того, что они настоящие - ее выдает дискриминатор. Сама генерация осуществляется функцией sample из src/training_model/sample.py. Присутсвует возможность скачать себе картинки на компьютер в формате архива. Также есть возможность оценить качество генерации, расчитав FID-score. Для этого надо пролистать страницу в самый низ и там будет кнопка. Эта функция использует AJAX и не обновляет страницу.
