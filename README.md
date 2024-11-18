Используемые библиотеки и технологии.
- python, javascript, pytorch, opencv, numpy, flask.

1. Описание проекта.
    Этот проект представляет собой удобный WEB-API, для взаимодействия с моделью, которая генерирует изображения ладоней (не берет из некоторой базы данных
а именно генерирует). Пользователь может создать сколько угодно картинок с ладонями человека всего за пару кликов.
    Основное применение такого инструмента состоит в быстром и дешевом формировании базы данных с изображениями ладоней, которая в дальнейшем может использоваться для обучения ML-модели для разных задач связанными с биометрией, такими как: выделение на руке линий, определение по двум ладоням, принадлежат ли они одному и тому же человеку, определение пола, возраста по изображениям ладоней и т.д.

2. Функционал сервиса.
    Планируется обучить модель, которая будет генерировать изображения ладоней. Обучение будет происходить на датасете, который есть в открытом доступе. После этого будет реализовано web-api, для взаимодействия с моделью. Пользователь будет иметь возможность сгенерировать сколько ему угодно изображений, оценить их правдоподобность с помощью другой модели прямо на сайте, скачать результаты. Генерация изображений будет быстрой в силу используемой архитектуры (об этом в следующей части).

3. Архитектура.
    a) Для генерации изображений используются генеративно-состязательные нейронные сети. Подход заключается в том, что в процессе обучения обучаются дые модели - дискриминатор и генератор. Генератор генерирует картинки из случайного шума, а дискриминатор учится отличать настоящие картинки от сгенерированных. Соответственно задача генератора в том, чтобы "обманывать" дискриминатор, а дискриминатора не обманываться. Дискриминатору во время одной эпохи подаются настоящие картинки и на них считается loss_real, а затем сгенерированные генератором изображения, считается loss_fake, после чего делается шаг обучения дискриминатора. Ошибка генератора это ошибка вердикта дискриминатора на сгенерированных данных. Дискриминатор и генератор реализованы как классы, наследующиеся от torch.nn.Module. Аттрибуты этих классов представляют собой последовательность слоев сети, методами же являются переопределенные __init__, forward. Также на этапе обучениядля улучшения качества датасета и обучения используется модель YOLO11, обученная детектировать ладони на изображении. Она используется для обрезки краев изображения. На этапе получения новых изображений, генератор из случайного шума генерирует новые картинки, и далее мы выбираем программно те, которым дискриминатор дал наибольшую вероятность быть настоящими. Обучение обернуто в класс Trainer, который реализует обучение, сохраняет модели и ошибки.
    б) Для frontend части будут использоваться шаблоны, написанные с помощью html, css, которые будут рендериться flask. Будет создан несложный интерфейс взаимодействия с моделью. Будут созданы несколько кнопок, каждая из которых будет отвечать за определенное действие.Будет форма, в которой пользователь может указать количество труб, которое он хочет сгенерировать, после этого нажав на кнопку модель начинает генерировать изображения и по завершению с бэкенда посылаются данные на сайт и отрисовываются. Далее пользователь нажав другую кнопку может автоматически оценить генерацию, получив рассчитанное значение FID (чем меньше это значение тем лучше, если оно меньше 100 при генерации примерно в 1000 картинок, то генерацию можно считать хорошей). Также пользователь будет иметь возможность просматривать картинки прямо на сайте, а также скачать их к себе на компьютер нажав одну кнопку.
   в) Бэкенд будет написан на python Flask. Получив запрос на генерацию картинок с клиента, сначала оттработает модель, картинки сохранятся в папку static/images. Затем эти картинки вместе с вероятностью их принадлежности классу реальных картинок будут отправлены на клиент и отрисованы. По нажатию на специальную кнопку будет возможность рассчитать FID score.
