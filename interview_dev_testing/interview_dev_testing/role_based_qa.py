ROLE_BASED_QA = {
    "ML Engineer": {
        "easy": [
            {"question": "What is overfitting in machine learning?", "answer": "Overfitting occurs when a model learns not only the underlying patterns but also the noise in the training data, leading to poor generalization."},
            {"question": "Can you name some common machine learning algorithms?", "answer": "Common algorithms include linear regression, decision trees, support vector machines (SVM), and neural networks."},
            {"question": "What is the difference between supervised and unsupervised learning?", "answer": "Supervised learning uses labeled data to predict outcomes, while unsupervised learning finds hidden patterns in unlabeled data."},
            {"question": "How would you explain a confusion matrix?", "answer": "A confusion matrix is used to evaluate the performance of classification models by showing the actual versus predicted classifications."},
            {"question": "What is the purpose of a validation set?", "answer": "The validation set is used to tune hyperparameters and prevent overfitting by evaluating the model's performance."},
            {"question": "Can you explain what a decision tree is?", "answer": "A decision tree is a flowchart-like model where data is split based on feature values to make predictions."},
            {"question": "What is gradient descent?", "answer": "Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating model parameters."},
            {"question": "What is the bias-variance tradeoff?", "answer": "The bias-variance tradeoff refers to the balance between underfitting (high bias) and overfitting (high variance) in a model."},
            {"question": "What is a feature in machine learning?", "answer": "A feature is an individual measurable property or characteristic used as input to a machine learning model."},
            {"question": "What does L1 and L2 regularization do?", "answer": "L1 regularization adds the absolute value of weights as a penalty, while L2 regularization adds the squared value of weights, helping to prevent overfitting."}
        ],
        "medium": [
            {"question": "How do you handle imbalanced datasets in classification problems?", "answer": "Techniques include resampling (over-sampling minority or under-sampling majority), using different metrics, or applying synthetic data generation methods like SMOTE."},
            {"question": "What is cross-validation, and why is it important?", "answer": "Cross-validation is a method for assessing model performance by splitting the data into multiple subsets to avoid overfitting."},
            {"question": "What is the difference between a generative and discriminative model?", "answer": "Generative models model the joint probability (P(X,Y)) and can generate new data instances, while discriminative models focus on P(Y|X) to classify data."},
            {"question": "How do you deal with missing data?", "answer": "Methods include imputation (mean, median, mode), removing missing data, or using models that handle missing values, such as decision trees."},
            {"question": "Can you explain the concept of ensemble learning?", "answer": "Ensemble learning combines multiple models to improve accuracy and robustness, using methods like bagging, boosting, or stacking."},
            {"question": "How do you evaluate the performance of a regression model?", "answer": "Metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared are commonly used."},
            {"question": "What is a ROC curve?", "answer": "A ROC curve plots the true positive rate (recall) against the false positive rate to assess the performance of a binary classifier."},
            {"question": "How does Principal Component Analysis (PCA) work?", "answer": "PCA reduces dimensionality by transforming the original features into new, uncorrelated variables called principal components."},
            {"question": "Can you explain the difference between bagging and boosting?", "answer": "Bagging builds multiple independent models and combines their predictions, while boosting builds models sequentially, where each new model corrects errors made by the previous one."},
            {"question": "What is the role of a kernel in SVM?", "answer": "A kernel function transforms the input data into a higher-dimensional space where it becomes easier to separate data points using a hyperplane."}
        ],
        "hard": [
            {"question": "Can you explain the backpropagation algorithm in deep learning?", "answer": "Backpropagation is a technique for updating the weights of a neural network by computing gradients of the loss function with respect to each weight using the chain rule."},
            {"question": "What are the advantages and disadvantages of using a convolutional neural network (CNN)?", "answer": "CNNs excel in image-related tasks due to local feature extraction but are computationally expensive and require large datasets."},
            {"question": "How does dropout work in neural networks?", "answer": "Dropout is a regularization technique where random neurons are deactivated during training to prevent overfitting."},
            {"question": "What is the Vanishing Gradient Problem, and how is it addressed?", "answer": "The Vanishing Gradient Problem occurs when gradients become too small for deep networks, preventing learning. Solutions include using ReLU activations or LSTM/GRU in recurrent networks."},
            {"question": "How do you handle exploding gradients?", "answer": "Techniques like gradient clipping or using normalization methods such as batch normalization can be applied to manage exploding gradients."},
            {"question": "What is transfer learning, and how would you apply it to deep learning models?", "answer": "Transfer learning involves using a model trained on one task as the starting point for a new task, often with fewer labeled examples."},
            {"question": "Can you explain the architecture of a Transformer model?", "answer": "Transformer models use self-attention mechanisms to capture dependencies between input tokens and have an encoder-decoder structure, unlike recurrent models."},
            {"question": "How do reinforcement learning algorithms differ from supervised learning?", "answer": "Reinforcement learning focuses on learning through rewards and penalties by interacting with an environment, while supervised learning relies on labeled data."},
            {"question": "What are generative adversarial networks (GANs)?", "answer": "GANs consist of two networks, a generator that creates fake data and a discriminator that distinguishes between real and fake data, trained adversarially."},
            {"question": "How do you implement custom loss functions in deep learning frameworks like TensorFlow or PyTorch?", "answer": "Custom loss functions can be implemented by defining the mathematical operation for the loss and using libraries' APIs like `tf.losses.Loss` in TensorFlow or defining functions in PyTorch."}
        ]
    },

    "Data Analyst": {
        "easy": [
            {"question": "What is a pivot table?", "answer": "A pivot table is a data summarization tool in Excel that is used to sort, aggregate, and organize data."},
            {"question": "What is a histogram?", "answer": "A histogram is a graphical representation of the distribution of numerical data."},
            {"question": "How do you calculate the mean of a dataset?", "answer": "The mean is calculated by summing all values and dividing by the number of data points."},
            {"question": "What is the difference between a bar chart and a pie chart?", "answer": "A bar chart represents categorical data with rectangular bars, while a pie chart shows proportions of a whole as slices."},
            {"question": "What is data wrangling?", "answer": "Data wrangling refers to the process of cleaning and transforming raw data into a usable format."},
            {"question": "What is a scatter plot used for?", "answer": "A scatter plot is used to observe the relationship between two continuous variables."},
            {"question": "How do you filter data in Excel or Google Sheets?", "answer": "You can filter data by using the filter option, allowing you to view only the rows that meet certain criteria."},
            {"question": "What is the median in a dataset?", "answer": "The median is the middle value in a dataset when arranged in ascending order."},
            {"question": "What does a correlation matrix show?", "answer": "A correlation matrix shows the correlation coefficients between variables in a dataset, indicating the strength and direction of relationships."},
            {"question": "How do you use the VLOOKUP function in Excel?", "answer": "The VLOOKUP function searches for a value in the leftmost column and returns a value in the same row from a specified column."}
        ],
        "medium": [
            {"question": "What is data normalization?", "answer": "Data normalization refers to adjusting values in a dataset to a common scale without distorting differences in ranges."},
            {"question": "How do you handle missing data?", "answer": "You can handle missing data by imputing values (e.g., mean, median), removing missing data, or flagging it for further analysis."},
            {"question": "What is the difference between correlation and causation?", "answer": "Correlation is when two variables move together, while causation implies that one variable causes the other to change."},
            {"question": "Can you explain hypothesis testing?", "answer": "Hypothesis testing is a statistical method that uses sample data to determine whether a hypothesis about a population parameter is true."},
            {"question": "How do you create a box plot, and what does it represent?", "answer": "A box plot represents the distribution of data through quartiles and highlights outliers."},
            {"question": "What is standard deviation, and why is it important?", "answer": "Standard deviation measures the spread or dispersion of a dataset and is crucial for understanding variability."},
            {"question": "Can you explain what an outlier is?", "answer": "An outlier is a data point that differs significantly from other observations, often affecting analysis results."},
            {"question": "How do you calculate the 95% confidence interval?", "answer": "The 95% confidence interval is calculated as the mean ± 1.96 times the standard error."},
            {"question": "What is the purpose of data aggregation?", "answer": "Data aggregation is the process of summarizing detailed data to a higher level for analysis."},
            {"question": "How do you perform a linear regression analysis in Excel?", "answer": "Linear regression in Excel can be performed by using the Data Analysis tool or the LINEST function."}
        ],
        "hard": [
            {"question": "Explain the differences between data mining and data analysis.", "answer": "Data mining involves discovering patterns in large datasets, while data analysis interprets data to inform decisions."},
            {"question": "Can you describe the SQL window functions?", "answer": "SQL window functions perform calculations across a set of table rows related to the current row without collapsing rows."},
            {"question": "How do you interpret a p-value in hypothesis testing?", "answer": "A p-value indicates the probability of observing the test results under the null hypothesis; a lower p-value suggests rejecting the null."},
            {"question": "What is the difference between k-means and hierarchical clustering?", "answer": "K-means is a centroid-based algorithm requiring a set number of clusters, while hierarchical clustering builds a tree of clusters."},
            {"question": "How would you analyze time series data?", "answer": "Time series data can be analyzed using methods like ARIMA, exponential smoothing, or by identifying trends and seasonality."},
            {"question": "How do you assess the quality of your data models?", "answer": "Model quality is assessed using metrics such as R-squared, Adjusted R-squared, RMSE, or using cross-validation techniques."},
            {"question": "What is the significance of R-squared in a regression analysis?", "answer": "R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variables."},
            {"question": "How do you deal with multicollinearity in regression?", "answer": "Multicollinearity can be addressed by removing highly correlated variables, using regularization, or performing PCA."},
            {"question": "What are the assumptions of ANOVA?", "answer": "ANOVA assumes independence of observations, normally distributed groups, and equal variances across groups."},
            {"question": "How do you apply machine learning to analyze large datasets?", "answer": "For large datasets, techniques like parallel processing, feature selection, and using distributed frameworks such as Apache Spark are applied."}
        ]
    },

    "Gen AI Developer": {
        "easy": [
            {"question": "What is a pre-trained model?", "answer": "A pre-trained model is a machine learning model trained on a large dataset and can be fine-tuned for specific tasks."},
            {"question": "Can you explain prompt engineering?", "answer": "Prompt engineering involves crafting inputs to guide a generative AI model to produce desired outputs."},
            {"question": "What is transfer learning?", "answer": "Transfer learning refers to applying a pre-trained model to a new but related task, allowing faster training."},
            {"question": "What is the difference between a generative model and a discriminative model?", "answer": "A generative model models the joint probability distribution (P(X,Y)), while a discriminative model focuses on conditional probability P(Y|X)."},
            {"question": "How does fine-tuning a model work?", "answer": "Fine-tuning a model involves further training a pre-trained model on a smaller, task-specific dataset to adapt it for a new task."},
            {"question": "What is a language model?", "answer": "A language model predicts the next word in a sentence or a sequence of words, often used in NLP tasks."},
            {"question": "What is the GPT model known for?", "answer": "GPT (Generative Pre-trained Transformer) is known for generating human-like text based on prompts."},
            {"question": "What is an API in the context of AI?", "answer": "An API (Application Programming Interface) allows developers to interact with machine learning models programmatically."},
            {"question": "How do embeddings work in NLP?", "answer": "Embeddings are vector representations of words that capture semantic relationships and can be used in downstream tasks."},
            {"question": "What is zero-shot learning?", "answer": "Zero-shot learning refers to a model's ability to correctly make predictions on unseen tasks without additional training."}
        ],
        "medium": [
            {"question": "How does a transformer architecture work?", "answer": "Transformer architecture uses self-attention to weigh the relevance of different words in a sequence, improving parallelization and performance in NLP."},
            {"question": "What is the difference between BERT and GPT models?", "answer": "BERT is a bidirectional transformer focused on masked word prediction, while GPT is unidirectional and specializes in text generation."},
            {"question": "What is tokenization in NLP?", "answer": "Tokenization is the process of breaking down text into smaller units (tokens) like words or subwords for processing by a model."},
            {"question": "How do attention mechanisms work in transformers?", "answer": "Attention mechanisms assign different importance weights to words in a sequence, allowing models to focus on the most relevant parts."},
            {"question": "What is beam search in sequence generation models?", "answer": "Beam search is a search algorithm that explores multiple possible sequences of words at each step to find the most likely output."},
            {"question": "How does few-shot learning improve model performance?", "answer": "Few-shot learning allows a model to learn new tasks with minimal data, leveraging prior knowledge from other tasks."},
            {"question": "What is RLHF (Reinforcement Learning with Human Feedback)?", "answer": "RLHF involves using human feedback to guide the training of a model, improving its alignment with desired outputs."},
            {"question": "How do you avoid hallucination in generative models?", "answer": "Avoiding hallucination involves using strong training data, refining prompts, and incorporating grounding mechanisms like retrieval-augmented generation."},
            {"question": "How does training on diverse datasets improve generalization in generative models?", "answer": "Training on diverse datasets exposes models to a wider range of patterns, improving their ability to generalize across different tasks."},
            {"question": "What are the trade-offs between model size and performance in LLMs?", "answer": "Larger models typically offer better performance but come at the cost of increased computation, memory, and latency, requiring efficient optimization strategies."}
        ],
        "hard": [
            {"question": "Can you explain the architecture of GPT models?", "answer": "GPT models are based on the Transformer architecture, using a decoder-only structure that relies on masked self-attention to predict the next word in a sequence."},
            {"question": "How do you optimize generative models for specific tasks?", "answer": "Generative models are optimized for specific tasks through fine-tuning on domain-specific data, hyperparameter tuning, and reinforcement learning."},
            {"question": "What is the impact of context length in language models?", "answer": "Context length determines how many previous tokens a model can consider when making predictions, with longer contexts typically yielding more coherent outputs."},
            {"question": "How does a mixture of experts (MoE) model work in large-scale AI?", "answer": "MoE models split input data across multiple expert models and select the most relevant expert for each input, improving efficiency and performance."},
            {"question": "How do you apply curriculum learning to fine-tune AI models?", "answer": "Curriculum learning involves training models on progressively harder tasks or data, helping them learn more effectively and improving generalization."},
            {"question": "What is the impact of pretraining data quality on generative models?", "answer": "The quality of pretraining data directly affects the accuracy and generalization of generative models, with diverse, clean data leading to better performance."},
            {"question": "What is the importance of decoder-only architectures in generative models?", "answer": "Decoder-only architectures allow models to focus solely on generating text by leveraging attention on previously generated tokens, making them well-suited for tasks like language modeling."},
            {"question": "How does retrieval-augmented generation (RAG) improve language model outputs?", "answer": "RAG improves model outputs by retrieving relevant documents or knowledge during generation, grounding predictions in factual information."},
            {"question": "How do you apply reinforcement learning in language models?", "answer": "Reinforcement learning can fine-tune language models by rewarding desirable outputs and penalizing incorrect or undesirable responses, improving model performance."},
            {"question": "What role does adversarial training play in improving generative model robustness?", "answer": "Adversarial training exposes models to perturbed or adversarial examples during training, making them more robust to adversarial attacks and unexpected inputs."}
        ]
    },

    "Web Developer": {
        "easy": [
            {"question": "What does HTML stand for?", "answer": "HTML stands for HyperText Markup Language, used to create the structure of web pages."},
            {"question": "What is CSS used for?", "answer": "CSS (Cascading Style Sheets) is used to style and layout web pages, controlling the design and appearance."},
            {"question": "What is JavaScript?", "answer": "JavaScript is a programming language used to make web pages interactive."},
            {"question": "What is the purpose of the <div> tag in HTML?", "answer": "The <div> tag is used to group and structure blocks of content in HTML."},
            {"question": "What is a responsive web design?", "answer": "Responsive web design allows web pages to look good on different devices by adjusting layout based on screen size."},
            {"question": "What is the difference between classes and IDs in CSS?", "answer": "Classes can be reused multiple times on a page, while IDs are unique and can only be used once per page."},
            {"question": "What is the purpose of the 'alt' attribute in an image tag?", "answer": "The 'alt' attribute provides alternative text for an image, which is useful for accessibility and SEO."},
            {"question": "What is a front-end framework?", "answer": "A front-end framework like React or Angular helps developers build user interfaces efficiently by providing reusable components."},
            {"question": "What is a CSS selector?", "answer": "A CSS selector is used to select and apply styles to HTML elements based on attributes, classes, or IDs."},
            {"question": "What is a hyperlink in HTML?", "answer": "A hyperlink is created using the <a> tag, allowing users to navigate to different web pages or resources."}
        ],
        "medium": [
            {"question": "What is the DOM?", "answer": "The DOM (Document Object Model) is a programming interface that represents the structure of a web page as a tree of objects."},
            {"question": "Can you explain the box model in CSS?", "answer": "The box model represents the layout of elements, including content, padding, border, and margin."},
            {"question": "What is Flexbox in CSS?", "answer": "Flexbox is a CSS layout module that allows for the creation of flexible, responsive layouts by controlling the alignment, direction, and spacing of items."},
            {"question": "What is the purpose of JavaScript events?", "answer": "JavaScript events allow developers to trigger specific functions or actions when a user interacts with elements on a web page (e.g., clicks, hovers)."},
            {"question": "What is an API, and how is it used in web development?", "answer": "An API (Application Programming Interface) allows web applications to interact with external systems, databases, or services."},
            {"question": "What is AJAX?", "answer": "AJAX (Asynchronous JavaScript and XML) allows web pages to be updated asynchronously by exchanging data with a web server in the background."},
            {"question": "What is the difference between GET and POST requests?", "answer": "GET requests retrieve data from a server, while POST requests send data to a server for processing."},
            {"question": "What is a CSS preprocessor?", "answer": "A CSS preprocessor (e.g., Sass, LESS) adds features like variables and mixins to CSS to make it more efficient to write and maintain."},
            {"question": "What is the purpose of media queries in CSS?", "answer": "Media queries allow developers to apply different CSS rules based on the device's screen size, orientation, or other characteristics."},
            {"question": "What is a single-page application (SPA)?", "answer": "An SPA is a web application that loads a single HTML page and dynamically updates content without reloading the entire page."}
        ],
        "hard": [
            {"question": "How does server-side rendering (SSR) differ from client-side rendering (CSR)?", "answer": "SSR renders web pages on the server and sends the fully rendered page to the client, while CSR renders content on the client side using JavaScript."},
            {"question": "Can you explain the differences between HTTP/1.1 and HTTP/2?", "answer": "HTTP/2 improves upon HTTP/1.1 by allowing multiplexing, header compression, and server push to enhance performance."},
            {"question": "What are web workers, and how are they used?", "answer": "Web workers allow JavaScript to run in the background, parallel to the main thread, improving the performance of complex operations."},
            {"question": "What is the purpose of Content Security Policy (CSP)?", "answer": "CSP is a security measure that helps prevent cross-site scripting (XSS) attacks by specifying which sources are allowed to load content."},
            {"question": "How do you optimize a website for performance?", "answer": "Website optimization involves techniques like minification, lazy loading, using CDNs, reducing HTTP requests, and optimizing images."},
            {"question": "What is WebSockets, and when would you use them?", "answer": "WebSockets enable real-time, bi-directional communication between a server and a client, often used in live chats or gaming applications."},
            {"question": "How does the 'async' and 'defer' attribute affect loading of JavaScript?", "answer": "'Async' loads JavaScript files asynchronously while the HTML parsing continues, while 'defer' loads them after the HTML parsing is complete."},
            {"question": "What are Service Workers, and how do they relate to Progressive Web Apps (PWAs)?", "answer": "Service Workers allow PWAs to cache assets, handle offline functionality, and manage background processes like push notifications."},
            {"question": "How does lazy loading work?", "answer": "Lazy loading delays the loading of non-critical resources (e.g., images) until they are needed, improving initial page load times."},
            {"question": "What is CORS (Cross-Origin Resource Sharing), and why is it important?", "answer": "CORS is a security feature that allows or restricts web pages from making requests to a domain other than the one serving the web page."}
        ]
    },

    "Android Developer": {
        "easy": [
            {"question": "What is Android?", "answer": "Android is an open-source mobile operating system developed by Google for smartphones, tablets, and other devices."},
            {"question": "What is an APK?", "answer": "APK (Android Package Kit) is the file format used to distribute and install applications on Android devices."},
            {"question": "What is an Activity in Android?", "answer": "An Activity is a single screen in an Android app that represents the user interface."},
            {"question": "What is the role of XML in Android development?", "answer": "XML is used to define the layout and UI components in Android apps."},
            {"question": "What is the AndroidManifest.xml file?", "answer": "AndroidManifest.xml is a configuration file that contains information about the app's components, permissions, and requirements."},
            {"question": "What is an Intent in Android?", "answer": "An Intent is a messaging object used to request an action from another app component."},
            {"question": "What is a RecyclerView in Android?", "answer": "RecyclerView is a flexible and efficient view for displaying a large set of data in a list or grid."},
            {"question": "What is the purpose of Gradle in Android development?", "answer": "Gradle is a build automation tool used to compile and build Android applications."},
            {"question": "What is Logcat in Android Studio?", "answer": "Logcat is a tool that displays log messages generated by the system and apps during development."},
            {"question": "What is the difference between dp and sp in Android?", "answer": "dp (density-independent pixels) is used for layout dimensions, while sp (scale-independent pixels) is used for font sizes."}
        ],
        "medium": [
            {"question": "What is the lifecycle of an Android Activity?", "answer": "The lifecycle includes states like onCreate(), onStart(), onResume(), onPause(), onStop(), and onDestroy(), which manage an activity's state."},
            {"question": "How do you handle screen orientation changes in Android?", "answer": "You can handle screen orientation changes by using the android:configChanges attribute or saving instance states using onSaveInstanceState()."},
            {"question": "What is the difference between Fragment and Activity?", "answer": "An Activity represents a single screen, while a Fragment is a reusable portion of the UI that can be embedded in Activities."},
            {"question": "What is ViewModel in Android?", "answer": "ViewModel is part of the Android Architecture Components that holds UI-related data in a lifecycle-conscious way."},
            {"question": "How do you implement data persistence in Android?", "answer": "Data persistence can be implemented using SharedPreferences, SQLite databases, Room, or files."},
            {"question": "What is the purpose of the ConstraintLayout?", "answer": "ConstraintLayout is a flexible layout that allows for complex positioning of UI elements by using constraints relative to other elements."},
            {"question": "What is Retrofit, and why is it used?", "answer": "Retrofit is a type-safe HTTP client for Android that simplifies communication with RESTful APIs."},
            {"question": "What is a BroadcastReceiver?", "answer": "A BroadcastReceiver is a component that listens for system-wide broadcast events like low battery or connectivity changes."},
            {"question": "How do you implement background tasks in Android?", "answer": "Background tasks can be implemented using AsyncTask, WorkManager, or IntentService."},
            {"question": "What is ProGuard, and why is it used?", "answer": "ProGuard is a tool that optimizes and obfuscates Android app code to reduce size and protect it from reverse engineering."}
        ],
        "hard": [
            {"question": "What is Dependency Injection, and how is it used in Android?", "answer": "Dependency Injection is a design pattern that provides objects that an app's components depend on. Dagger or Hilt can be used for DI in Android."},
            {"question": "How does Android handle memory management?", "answer": "Android uses a garbage collector to free up memory by removing unused objects, but developers should manage memory leaks through efficient resource handling."},
            {"question": "What is the difference between Parcelable and Serializable?", "answer": "Parcelable is more efficient than Serializable in Android for passing data between activities or fragments."},
            {"question": "What is LiveData, and how is it different from MutableLiveData?", "answer": "LiveData is an observable data holder that respects the lifecycle of components, while MutableLiveData allows modification of data."},
            {"question": "What is the purpose of the WorkManager API?", "answer": "WorkManager is used for scheduling deferrable and guaranteed background tasks in Android that should run even if the app is closed or the device is rebooted."},
            {"question": "How does Room Database differ from SQLite?", "answer": "Room is an abstraction layer over SQLite that simplifies database access and provides compile-time verification of queries."},
            {"question": "What are the advantages of Kotlin over Java in Android development?", "answer": "Kotlin offers features like null safety, extension functions, and coroutines, leading to more concise and safe code compared to Java."},
            {"question": "How do you implement app architecture in Android using MVVM?", "answer": "MVVM architecture separates concerns by organizing code into Model, View, and ViewModel layers, improving maintainability and testability."},
            {"question": "What is Jetpack Compose?", "answer": "Jetpack Compose is Android’s modern UI toolkit that simplifies UI development with declarative programming, replacing XML-based layouts."},
            {"question": "What are Coroutines in Kotlin, and how are they used in Android?", "answer": "Coroutines are lightweight threads in Kotlin used to perform asynchronous tasks efficiently without blocking the main thread."}
        ]
    },

    "Blockchain Developer": {
        "easy": [
            {"question": "What is a blockchain?", "answer": "Blockchain is a distributed digital ledger that records transactions across multiple computers in a secure, transparent, and immutable way."},
            {"question": "What is a cryptocurrency?", "answer": "Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates on a decentralized network, typically a blockchain."},
            {"question": "What is Bitcoin?", "answer": "Bitcoin is the first decentralized cryptocurrency, created in 2009 by an unknown person or group using the name Satoshi Nakamoto."},
            {"question": "What is a smart contract?", "answer": "A smart contract is a self-executing contract with the terms of the agreement directly written into code, running on a blockchain."},
            {"question": "What is Ethereum?", "answer": "Ethereum is a decentralized platform that allows developers to build and deploy smart contracts and decentralized applications (dApps)."},
            {"question": "What is a public blockchain?", "answer": "A public blockchain is an open network that anyone can join, view, and participate in without restrictions."},
            {"question": "What is the difference between a coin and a token?", "answer": "A coin operates on its own blockchain (e.g., Bitcoin), while a token is built on top of an existing blockchain (e.g., Ethereum tokens)."},
            {"question": "What is mining in blockchain?", "answer": "Mining is the process of validating transactions and adding them to the blockchain, often requiring computational work to solve complex problems."},
            {"question": "What is a private key in blockchain?", "answer": "A private key is a secret number used in cryptography, allowing users to sign transactions and access their blockchain assets."},
            {"question": "What is a wallet in blockchain?", "answer": "A wallet is a digital tool that stores a user's private and public keys and allows them to send and receive cryptocurrencies."}
        ],
        "medium": [
            {"question": "What is a consensus mechanism?", "answer": "A consensus mechanism is a protocol used by blockchain networks to agree on the validity of transactions and to secure the network, such as Proof of Work or Proof of Stake."},
            {"question": "What is Proof of Work (PoW)?", "answer": "Proof of Work is a consensus algorithm used in blockchain networks, requiring participants to solve complex mathematical puzzles to validate transactions."},
            {"question": "What is Proof of Stake (PoS)?", "answer": "Proof of Stake is a consensus mechanism where participants validate transactions based on the number of coins they hold, reducing energy consumption compared to PoW."},
            {"question": "What is a dApp?", "answer": "A dApp (decentralized application) is a digital application that runs on a blockchain or peer-to-peer network instead of centralized servers."},
            {"question": "What is gas in Ethereum?", "answer": "Gas is the unit used to measure the amount of computational effort required to perform operations on the Ethereum blockchain."},
            {"question": "What is a fork in blockchain?", "answer": "A fork occurs when a blockchain diverges into two separate paths, often due to a protocol update or disagreement in the community."},
            {"question": "What is a DAO (Decentralized Autonomous Organization)?", "answer": "A DAO is an organization represented by rules encoded as a computer program that is transparent, controlled by members, and operates without central authority."},
            {"question": "What are Layer 1 and Layer 2 solutions in blockchain?", "answer": "Layer 1 refers to the base blockchain network (e.g., Ethereum), while Layer 2 solutions are protocols built on top to improve scalability (e.g., Lightning Network)."},
            {"question": "What is sharding in blockchain?", "answer": "Sharding is a technique used to split a blockchain network into smaller partitions, called shards, to improve scalability and transaction throughput."},
            {"question": "What is a block in blockchain?", "answer": "A block is a record of transactions that is permanently recorded on the blockchain and linked to other blocks to form a chain."}
        ],
        "hard": [
            {"question": "How does the Byzantine Generals Problem relate to blockchain?", "answer": "The Byzantine Generals Problem is a game theory concept that blockchain solves using consensus algorithms, ensuring agreement among decentralized nodes even with malicious actors."},
            {"question": "What is the Ethereum Virtual Machine (EVM)?", "answer": "The EVM is a decentralized, Turing-complete virtual machine that executes smart contracts on the Ethereum blockchain."},
            {"question": "How does elliptic curve cryptography work in blockchain?", "answer": "Elliptic curve cryptography (ECC) is used in blockchain for secure encryption and signing of transactions, providing high levels of security with smaller keys."},
            {"question": "What is the difference between a soft fork and a hard fork?", "answer": "A soft fork is a backward-compatible upgrade to the blockchain, while a hard fork results in a permanent divergence requiring nodes to upgrade."},
            {"question": "What is zk-SNARK, and how is it used in blockchain?", "answer": "zk-SNARK (Zero-Knowledge Succinct Non-Interactive Argument of Knowledge) allows one party to prove possession of information without revealing it, enhancing privacy in blockchain."},
            {"question": "What are Merkle trees, and how are they used in blockchain?", "answer": "Merkle trees are data structures used to efficiently and securely verify the integrity of transactions in a blockchain."},
            {"question": "What is a cross-chain bridge?", "answer": "A cross-chain bridge allows tokens and assets to be transferred between different blockchain networks, improving interoperability."},
            {"question": "What is a hash function, and why is it important in blockchain?", "answer": "A hash function takes input data and produces a fixed-size string of characters, which is essential for securing blockchain transactions and creating block links."},
            {"question": "What is staking in blockchain?", "answer": "Staking involves locking up a certain amount of cryptocurrency to participate in the network's consensus mechanism, earning rewards in return."},
            {"question": "What is the role of smart contract auditing in blockchain?", "answer": "Smart contract auditing involves reviewing the code of smart contracts for vulnerabilities to ensure security and prevent potential exploits."}
        ]
    },

    "Django Developer": {
        "easy": [
            {"question": "What is Django?", "answer": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design."},
            {"question": "What is a Django model?", "answer": "A Django model is a Python class that maps to a database table, used to define the structure of the database schema."},
            {"question": "What is the purpose of 'urls.py' in Django?", "answer": "The 'urls.py' file in Django defines the URL patterns that route requests to the appropriate views."},
            {"question": "What is Django ORM?", "answer": "Django ORM (Object-Relational Mapping) allows developers to interact with the database using Python objects instead of SQL."},
            {"question": "What is a Django view?", "answer": "A Django view is a Python function or class that processes a request and returns a response, usually rendering an HTML template."},
            {"question": "What is the purpose of the 'settings.py' file?", "answer": "'settings.py' contains all the configuration and settings for a Django project, such as database configuration and static file settings."},
            {"question": "What is a Django template?", "answer": "A Django template is an HTML file mixed with Django Template Language (DTL), used to dynamically generate HTML output."},
            {"question": "What is 'manage.py' in Django?", "answer": "'manage.py' is a command-line utility that lets you interact with your Django project, such as running the development server or migrating the database."},
            {"question": "What is the difference between 'render()' and 'HttpResponse()' in Django?", "answer": "'render()' is used to return a template along with context data, while 'HttpResponse()' returns raw HTTP response data."},
            {"question": "What is the purpose of 'migrations' in Django?", "answer": "Migrations in Django are used to propagate changes made to models (such as adding or altering fields) into the database schema."}
        ],
        "medium": [
            {"question": "What is the use of middleware in Django?", "answer": "Middleware in Django is a framework that processes requests and responses. It sits between the request/response process to perform various tasks like authentication, session management, etc."},
            {"question": "How do you perform form validation in Django?", "answer": "Form validation in Django can be performed using Django forms, where you can define validation rules and handle errors using the 'clean' methods."},
            {"question": "What are Django signals?", "answer": "Django signals allow certain senders to notify receivers when specific actions have occurred, such as saving or deleting an object."},
            {"question": "How can you serve static files in Django?", "answer": "Static files like CSS, JavaScript, and images are served in Django by defining 'STATIC_URL' and 'STATICFILES_DIRS' in the settings file."},
            {"question": "What is the difference between 'select_related()' and 'prefetch_related()' in Django?", "answer": "'select_related()' performs a single SQL query for related objects in a foreign key relationship, while 'prefetch_related()' performs separate queries for many-to-many or reverse foreign key relationships."},
            {"question": "What is the purpose of 'Context Processors' in Django?", "answer": "Context processors in Django add variables to the context of templates globally, making them available to all templates."},
            {"question": "What is Django REST Framework (DRF)?", "answer": "Django REST Framework is a powerful toolkit for building Web APIs in Django, providing features like serialization, authentication, and viewsets."},
            {"question": "How do you secure a Django application?", "answer": "Securing a Django application includes setting strong passwords, using HTTPS, enabling CSRF protection, and avoiding SQL injections using ORM."},
            {"question": "What are Django Class-Based Views (CBVs)?", "answer": "Django CBVs are views that use Python classes instead of functions, offering more reusable and modular views."},
            {"question": "What is the purpose of Django 'Admin'?", "answer": "Django Admin is a built-in interface for managing application data through an easy-to-use graphical interface, allowing developers to manage models and users."}
        ],
        "hard": [
            {"question": "What is the N+1 query problem in Django, and how can you avoid it?", "answer": "The N+1 query problem happens when a query is made for a related object inside a loop, leading to multiple queries. It can be avoided by using 'select_related()' or 'prefetch_related()'."},
            {"question": "What is Celery, and how is it used in Django?", "answer": "Celery is a distributed task queue used for handling asynchronous tasks in Django, often for background processing or scheduling tasks."},
            {"question": "How does Django handle database transactions?", "answer": "Django uses the 'transaction' module to handle database transactions, ensuring that database operations are atomic and can be rolled back if necessary."},
            {"question": "How would you implement user authentication in Django?", "answer": "Django provides a built-in authentication system, including user registration, login, and password management. Custom authentication backends can also be implemented."},
            {"question": "What is the purpose of caching in Django?", "answer": "Caching in Django improves performance by storing frequently accessed data in memory, reducing database hits and speeding up page load times."},
            {"question": "How do you create a custom template filter in Django?", "answer": "A custom template filter in Django can be created by writing a Python function and registering it with 'register.filter' in the 'templatetags' module."},
            {"question": "What is the difference between 'REST' and 'GraphQL' in Django?", "answer": "REST provides structured, predefined endpoints for CRUD operations, while GraphQL allows clients to request only the specific data they need in a single query."},
            {"question": "What is the role of the 'Django Migrations Framework'?", "answer": "Django migrations are used to evolve your database schema over time, ensuring database consistency and allowing developers to version control schema changes."},
            {"question": "What are Django Custom Managers?", "answer": "Custom Managers in Django are used to add custom query methods to models by extending the default manager or creating new ones."},
            {"question": "How does Django handle security vulnerabilities like CSRF, XSS, and SQL Injection?", "answer": "Django provides CSRF tokens for forms, auto-escaping to prevent XSS, and ORM queries to prevent SQL injection."}
        ]
    },

    "DevOps": {
        "easy": [
            {"question": "What is DevOps?", "answer": "DevOps is a set of practices that combines software development (Dev) and IT operations (Ops) to shorten the software development lifecycle."},
            {"question": "What is Continuous Integration (CI)?", "answer": "Continuous Integration is the practice of frequently integrating code changes into a shared repository, followed by automated testing."},
            {"question": "What is a version control system?", "answer": "A version control system (VCS) is a tool that helps manage changes to code over time, with Git being a popular example."},
            {"question": "What is Continuous Delivery (CD)?", "answer": "Continuous Delivery is an approach where code changes are automatically tested and prepared for release to production environments."},
            {"question": "What is Infrastructure as Code (IaC)?", "answer": "Infrastructure as Code is the practice of managing and provisioning computing infrastructure using machine-readable configuration files."},
            {"question": "What is a Docker container?", "answer": "A Docker container is a lightweight, standalone package of software that includes everything needed to run an application."},
            {"question": "What is Kubernetes?", "answer": "Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications."},
            {"question": "What is Jenkins?", "answer": "Jenkins is an open-source automation server used for continuous integration and delivery (CI/CD) pipelines."},
            {"question": "What is Git?", "answer": "Git is a distributed version control system that tracks changes in source code during software development."},
            {"question": "What is the difference between a container and a virtual machine?", "answer": "Containers share the host system's kernel and resources, while virtual machines include a full operating system and require more resources."}
        ],
        "medium": [
            {"question": "What is the purpose of CI/CD pipelines?", "answer": "CI/CD pipelines automate the processes of building, testing, and deploying applications, ensuring continuous integration and continuous delivery."},
            {"question": "What are the differences between Docker and Kubernetes?", "answer": "Docker is used for containerization, while Kubernetes is used for container orchestration, managing clusters of containers at scale."},
            {"question": "What is load balancing, and why is it important?", "answer": "Load balancing distributes incoming traffic across multiple servers to ensure no single server is overwhelmed, improving availability and reliability."},
            {"question": "What is a microservice architecture?", "answer": "Microservice architecture breaks down a large application into smaller, loosely coupled services that can be developed, deployed, and scaled independently."},
            {"question": "What is a reverse proxy?", "answer": "A reverse proxy sits between clients and servers, forwarding client requests to the appropriate server and providing features like load balancing and SSL termination."},
            {"question": "What is Prometheus, and how is it used in DevOps?", "answer": "Prometheus is an open-source monitoring and alerting toolkit used to collect and record metrics in real-time for applications and infrastructure."},
            {"question": "What are blue-green deployments?", "answer": "Blue-green deployment is a technique where two identical environments (blue and green) are used to release new versions of an application without downtime."},
            {"question": "What is a service mesh?", "answer": "A service mesh is a dedicated infrastructure layer for controlling service-to-service communication, providing features like load balancing and observability."},
            {"question": "What is Terraform?", "answer": "Terraform is an open-source IaC tool that allows you to define, provision, and manage infrastructure across different cloud platforms."},
            {"question": "What is the purpose of monitoring and logging in DevOps?", "answer": "Monitoring tracks the health of systems and applications in real-time, while logging records events and errors to aid in troubleshooting."}
        ],
        "hard": [
            {"question": "What is the difference between Continuous Delivery and Continuous Deployment?", "answer": "In Continuous Delivery, code is automatically tested and prepared for release but requires manual approval to deploy, while Continuous Deployment fully automates the release process."},
            {"question": "How does Kubernetes handle scaling?", "answer": "Kubernetes scales applications by automatically adding or removing containers based on resource usage and demand, using horizontal pod autoscaling."},
            {"question": "What is Helm in Kubernetes?", "answer": "Helm is a package manager for Kubernetes that helps in defining, installing, and managing Kubernetes applications using 'Helm Charts.'"},
            {"question": "How do you handle secrets in a DevOps environment?", "answer": "Secrets management can be handled using tools like HashiCorp Vault, AWS Secrets Manager, or Kubernetes Secrets to securely store sensitive data."},
            {"question": "What is Canary Deployment?", "answer": "Canary deployment is a strategy where a new version of an application is gradually rolled out to a subset of users before full deployment."},
            {"question": "How do you handle configuration management in DevOps?", "answer": "Configuration management tools like Ansible, Puppet, or Chef are used to automate and maintain consistency of system configurations across environments."},
            {"question": "What are serverless architectures?", "answer": "Serverless architectures allow developers to build and run applications without managing the underlying infrastructure, relying on cloud services like AWS Lambda or Azure Functions."},
            {"question": "What is the importance of observability in DevOps?", "answer": "Observability refers to the ability to monitor, measure, and understand what is happening in a system based on the data it produces, crucial for maintaining performance and availability."},
            {"question": "What is a pipeline as code?", "answer": "Pipeline as code refers to defining the CI/CD pipeline in code using tools like Jenkins, GitLab CI, or CircleCI, allowing version control and collaboration."},
            {"question": "What is a multi-cloud strategy?", "answer": "A multi-cloud strategy involves using services from multiple cloud providers to avoid vendor lock-in, improve reliability, and optimize cost."}
        ]
    },
    "Cybersecurity Engineer": {
        "easy": [
            {"question": "What is a firewall?", "answer": "A firewall is a network security device that monitors and controls incoming and outgoing network traffic based on predetermined security rules."},
            {"question": "What is the difference between symmetric and asymmetric encryption?", "answer": "Symmetric encryption uses the same key for encryption and decryption, while asymmetric encryption uses a public and a private key pair."},
            {"question": "What is phishing?", "answer": "Phishing is a cyberattack method that uses disguised email or messages to trick people into revealing personal information or credentials."},
            {"question": "What is two-factor authentication?", "answer": "Two-factor authentication is a security process that requires two different authentication factors to verify a user's identity."},
            {"question": "What is malware?", "answer": "Malware is malicious software designed to harm, exploit, or take control of computer systems without the user's knowledge."},
            {"question": "What is the purpose of encryption?", "answer": "Encryption is used to protect data by converting it into a coded form that can only be deciphered by authorized parties with the correct decryption key."},
            {"question": "What is a DDoS attack?", "answer": "A Distributed Denial of Service (DDoS) attack involves overwhelming a server or network with traffic from multiple sources, causing it to crash."},
            {"question": "What is a VPN?", "answer": "A Virtual Private Network (VPN) is a service that encrypts your internet connection and hides your IP address to enhance privacy and security."},
            {"question": "What is the CIA triad in cybersecurity?", "answer": "The CIA triad stands for Confidentiality, Integrity, and Availability, representing the three core principles of cybersecurity."},
            {"question": "What is an antivirus program?", "answer": "An antivirus program is software designed to detect, prevent, and remove malicious software like viruses, worms, and Trojans from computers."}
        ],
        "medium": [
            {"question": "What is a man-in-the-middle (MITM) attack?", "answer": "A MITM attack is a type of cyberattack where the attacker secretly intercepts and potentially alters the communication between two parties."},
            {"question": "How does Public Key Infrastructure (PKI) work?", "answer": "PKI is a system that uses digital certificates to authenticate the identity of users, devices, or services, securing communications over the internet."},
            {"question": "What is an intrusion detection system (IDS)?", "answer": "An IDS is a device or software that monitors network traffic for suspicious activity and alerts administrators to potential threats."},
            {"question": "What is the difference between a virus and a worm?", "answer": "A virus requires a host program to spread, while a worm is a standalone program that can spread without human interaction."},
            {"question": "What is a vulnerability assessment?", "answer": "A vulnerability assessment is the process of identifying, quantifying, and prioritizing vulnerabilities in a system to minimize risks."},
            {"question": "What is cross-site scripting (XSS)?", "answer": "XSS is a vulnerability that allows attackers to inject malicious scripts into web pages viewed by other users."},
            {"question": "What is a zero-day vulnerability?", "answer": "A zero-day vulnerability is a software flaw that is unknown to the vendor and for which no patch is available, making it highly exploitable."},
            {"question": "What is the role of a Security Operations Center (SOC)?", "answer": "A SOC is a centralized unit that monitors and responds to security incidents in real-time, protecting the organization’s assets."},
            {"question": "How does encryption differ from hashing?", "answer": "Encryption is reversible and used to protect data, while hashing is one-way and used to verify data integrity."},
            {"question": "What is multi-factor authentication (MFA)?", "answer": "MFA is a security measure that requires multiple forms of verification, such as a password and a fingerprint, to authenticate a user."}
        ],
        "hard": [
            {"question": "How do you mitigate a buffer overflow attack?", "answer": "Buffer overflow attacks can be mitigated by using secure coding practices, input validation, and memory protection techniques such as Address Space Layout Randomization (ASLR)."},
            {"question": "What is the role of Security Information and Event Management (SIEM)?", "answer": "SIEM tools collect and analyze security data from across the network to detect threats, providing real-time monitoring and analysis of events."},
            {"question": "What is Advanced Persistent Threat (APT)?", "answer": "An APT is a prolonged and targeted cyberattack where an intruder gains unauthorized access and remains undetected for an extended period."},
            {"question": "How does SSL/TLS work?", "answer": "SSL/TLS encrypts data transmitted over the internet, ensuring secure communication between clients and servers using certificates for authentication."},
            {"question": "What is the difference between IDS and IPS?", "answer": "An Intrusion Detection System (IDS) detects and alerts administrators of suspicious activity, while an Intrusion Prevention System (IPS) actively blocks threats."},
            {"question": "What is the principle of least privilege?", "answer": "The principle of least privilege states that users should be granted the minimum level of access required to perform their job, reducing the risk of abuse or attacks."},
            {"question": "What are honey pots?", "answer": "Honey pots are decoy systems or services designed to lure attackers away from real targets and monitor their behavior."},
            {"question": "What is Data Loss Prevention (DLP)?", "answer": "DLP is a strategy for preventing the unauthorized disclosure or transfer of sensitive data, typically using monitoring, encryption, and access controls."},
            {"question": "How do you secure API communications?", "answer": "API communications can be secured using authentication, encryption, rate limiting, and monitoring for unusual activity."},
            {"question": "What is penetration testing?", "answer": "Penetration testing involves simulating a cyberattack on a system or network to find vulnerabilities that could be exploited by malicious actors."}
        ]
    },

    "Cloud Engineer": {
        "easy": [
            {"question": "What is cloud computing?", "answer": "Cloud computing is the delivery of computing services like servers, storage, databases, networking, and software over the internet."},
            {"question": "What are the main types of cloud computing services?", "answer": "The main types of cloud computing services are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS)."},
            {"question": "What is the difference between public, private, and hybrid clouds?", "answer": "Public clouds are operated by third-party providers, private clouds are dedicated to a single organization, and hybrid clouds combine both."},
            {"question": "What is AWS?", "answer": "Amazon Web Services (AWS) is a comprehensive cloud computing platform offered by Amazon, providing services such as storage, networking, and databases."},
            {"question": "What is the purpose of cloud storage?", "answer": "Cloud storage allows users to store data on remote servers that can be accessed from anywhere via the internet."},
            {"question": "What is virtualization in cloud computing?", "answer": "Virtualization is the process of creating virtual instances of resources like servers or storage devices, making them accessible over the cloud."},
            {"question": "What is auto-scaling in the cloud?", "answer": "Auto-scaling is a feature that automatically adjusts the number of compute resources to match the demand."},
            {"question": "What is an S3 bucket in AWS?", "answer": "An S3 bucket is a storage container in AWS S3 where users can store, retrieve, and manage data files."},
            {"question": "What is a Virtual Private Cloud (VPC)?", "answer": "A VPC is a virtual network within a public cloud, offering greater control and isolation over cloud resources."},
            {"question": "What is cloud elasticity?", "answer": "Cloud elasticity refers to the ability of a cloud service to automatically scale up or down based on demand."}
        ],
        "medium": [
            {"question": "What is serverless computing?", "answer": "Serverless computing allows developers to build applications without managing infrastructure, where the cloud provider dynamically manages server resources."},
            {"question": "What is a multi-cloud strategy?", "answer": "A multi-cloud strategy involves using multiple cloud services from different providers to avoid vendor lock-in and improve redundancy."},
            {"question": "What is the purpose of an API gateway in cloud architecture?", "answer": "An API gateway acts as an entry point for API requests, handling request routing, authentication, and throttling."},
            {"question": "What are the benefits of containerization in the cloud?", "answer": "Containerization enables faster deployment, scalability, and efficient resource management by isolating applications and their dependencies."},
            {"question": "What is Infrastructure as Code (IaC)?", "answer": "IaC allows cloud infrastructure to be managed and provisioned using code, enabling automation and version control."},
            {"question": "What is the difference between scaling vertically and scaling horizontally?", "answer": "Scaling vertically adds more resources to a single instance, while scaling horizontally adds more instances to handle increased load."},
            {"question": "What is Azure Blob Storage?", "answer": "Azure Blob Storage is Microsoft's object storage solution for storing large amounts of unstructured data in the cloud."},
            {"question": "What is CloudFormation in AWS?", "answer": "AWS CloudFormation is a service that allows users to define and provision AWS infrastructure using templates."},
            {"question": "How does cloud cost optimization work?", "answer": "Cloud cost optimization involves managing and reducing cloud expenses by rightsizing resources, using reserved instances, and leveraging cost monitoring tools."},
            {"question": "What is a Kubernetes cluster?", "answer": "A Kubernetes cluster is a set of nodes that run containerized applications, managed and orchestrated by Kubernetes."}
        ],
        "hard": [
            {"question": "What is the difference between AWS Lambda and EC2?", "answer": "AWS Lambda is a serverless computing service that runs code in response to events, while EC2 provides resizable compute capacity for applications."},
            {"question": "What is a cloud-native application?", "answer": "Cloud-native applications are built to take full advantage of cloud computing, using microservices, containers, and dynamic orchestration."},
            {"question": "What is the CAP theorem in cloud computing?", "answer": "The CAP theorem states that in a distributed data system, it is impossible to achieve Consistency, Availability, and Partition tolerance simultaneously."},
            {"question": "How do you secure data in the cloud?", "answer": "Data security in the cloud involves encryption, access controls, monitoring, and implementing identity and access management (IAM) policies."},
            {"question": "What is cloud orchestration?", "answer": "Cloud orchestration automates the management of complex cloud environments, including the deployment, configuration, and coordination of services."},
            {"question": "What is a cloud CDN?", "answer": "A Cloud Content Delivery Network (CDN) distributes content geographically to improve access speed and reduce latency."},
            {"question": "What are CloudFormation templates?", "answer": "CloudFormation templates are declarative JSON or YAML files used to define and provision AWS infrastructure."},
            {"question": "What is the shared responsibility model in cloud computing?", "answer": "The shared responsibility model divides security responsibilities between the cloud provider (infrastructure) and the customer (data, applications, and configuration)."},
            {"question": "How does hybrid cloud work?", "answer": "A hybrid cloud integrates on-premises infrastructure with public and private clouds, enabling data and application portability."},
            {"question": "What is cloud bursting?", "answer": "Cloud bursting is a hybrid cloud strategy where an application runs in a private cloud or data center and bursts into a public cloud when demand spikes."}
        ]
    },

    "Systems Engineer": {
        "easy": [
            {"question": "What is the role of a systems engineer?", "answer": "A systems engineer is responsible for designing, implementing, and managing the infrastructure that supports an organization's IT operations."},
            {"question": "What is an operating system?", "answer": "An operating system (OS) is software that manages hardware and software resources on a computer, providing a user interface and platform for applications."},
            {"question": "What is RAID, and why is it used?", "answer": "RAID (Redundant Array of Independent Disks) is a data storage virtualization technology used to improve performance and redundancy."},
            {"question": "What is a virtual machine (VM)?", "answer": "A virtual machine is an emulation of a computer system, allowing multiple operating systems to run on a single physical machine."},
            {"question": "What is the difference between IPv4 and IPv6?", "answer": "IPv4 uses 32-bit addresses, while IPv6 uses 128-bit addresses, offering a larger address space and better security features."},
            {"question": "What is network latency?", "answer": "Network latency refers to the time it takes for data to travel from one point to another over a network."},
            {"question": "What is the difference between TCP and UDP?", "answer": "TCP is a connection-oriented protocol that ensures reliable data transmission, while UDP is a connectionless protocol used for faster, less reliable transmissions."},
            {"question": "What is DNS?", "answer": "DNS (Domain Name System) is the system that translates human-readable domain names (like google.com) into IP addresses that computers use to identify each other on the network."},
            {"question": "What is a hypervisor?", "answer": "A hypervisor is software that creates and runs virtual machines by managing hardware resources and allowing multiple VMs to run on a single physical host."},
            {"question": "What is a server?", "answer": "A server is a computer that provides data, services, or programs to other computers (clients) over a network."}
        ],
        "medium": [
            {"question": "What is load balancing?", "answer": "Load balancing is the process of distributing incoming network traffic across multiple servers to ensure no single server becomes overwhelmed."},
            {"question": "What is the purpose of Active Directory?", "answer": "Active Directory is a Microsoft service used to manage user accounts, devices, and security policies in a networked environment."},
            {"question": "What is the difference between NAS and SAN?", "answer": "NAS (Network Attached Storage) is a file-level storage system, while SAN (Storage Area Network) is a block-level storage system."},
            {"question": "What is network segmentation?", "answer": "Network segmentation is the practice of dividing a network into smaller subnetworks to improve security and performance."},
            {"question": "What is SNMP?", "answer": "SNMP (Simple Network Management Protocol) is a protocol used to monitor and manage network devices like routers, switches, and servers."},
            {"question": "What is the role of a systems engineer in disaster recovery?", "answer": "Systems engineers are responsible for implementing and maintaining disaster recovery plans to ensure business continuity in case of a failure or disaster."},
            {"question": "What is the difference between a switch and a router?", "answer": "A switch connects devices within a network and routes data based on MAC addresses, while a router connects different networks and routes data based on IP addresses."},
            {"question": "What is virtualization, and why is it important?", "answer": "Virtualization is the process of creating virtual instances of physical resources, improving resource utilization, scalability, and isolation."},
            {"question": "What is the OSI model?", "answer": "The OSI (Open Systems Interconnection) model is a conceptual framework that standardizes network communication functions across seven layers."},
            {"question": "What is network redundancy?", "answer": "Network redundancy involves duplicating critical network components or paths to ensure availability in case of a failure."}
        ],
        "hard": [
            {"question": "What is the difference between IPv6 stateless and stateful autoconfiguration?", "answer": "Stateless autoconfiguration allows devices to configure their IP addresses without a DHCP server, while stateful autoconfiguration requires a DHCP server for address assignment."},
            {"question": "What is BGP, and how is it used in networking?", "answer": "BGP (Border Gateway Protocol) is a routing protocol used to exchange routing information between autonomous systems on the internet."},
            {"question": "How does high availability (HA) work in a data center?", "answer": "High availability is achieved through redundant infrastructure, failover systems, and load balancing to ensure continuous operation during failures."},
            {"question": "What is the difference between synchronous and asynchronous replication?", "answer": "Synchronous replication ensures data is written to both the primary and secondary site at the same time, while asynchronous replication allows some delay between the two."},
            {"question": "How do you implement disaster recovery in a virtualized environment?", "answer": "Disaster recovery in virtualized environments can be implemented using techniques like replication, snapshots, and failover clustering."},
            {"question": "What is SDN (Software-Defined Networking)?", "answer": "SDN is an approach to network management that enables dynamic, programmatically efficient configuration of network resources using software."},
            {"question": "What is the purpose of network traffic monitoring?", "answer": "Network traffic monitoring helps detect anomalies, measure performance, and troubleshoot network issues by analyzing the data flow across the network."},
            {"question": "What is VXLAN, and why is it used?", "answer": "VXLAN (Virtual Extensible LAN) is a network virtualization technology used to extend Layer 2 networks across Layer 3 infrastructure, providing greater scalability."},
            {"question": "What is clustering, and why is it important in systems engineering?", "answer": "Clustering is the practice of connecting multiple servers to work together as a single system, improving scalability, availability, and reliability."},
            {"question": "What is the difference between Fibre Channel and iSCSI?", "answer": "Fibre Channel is a high-speed network technology used for SANs, while iSCSI uses IP networks to transfer data, making it more cost-effective."}
        ]
    }
}




