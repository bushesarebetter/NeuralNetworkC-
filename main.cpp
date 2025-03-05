    #include <iostream>
    #include <vector>
    #include <random>
    #include <chrono>
    #include <thread>
    #include <future>
    #include <numeric>
    #include <algorithm>
    #include <sstream>
    #include <fstream>
    #include <istream>


    template<typename T>
    T getRandom(T min, T max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);
        return dist(gen);
    }

    
    template<typename T>
    class Matrix {
    private:
        std::vector<size_t> dimensions;
        std::vector<T> data;
        
        size_t calculateIndex(const std::vector<size_t>& indices) const {
            size_t index = 0;
            size_t multiplier = 1;
            
            for (int i = dimensions.size() - 1; i >= 0; --i) {
                index += indices[i] * multiplier;
                multiplier *= dimensions[i];
            }
            
            return index;
        }
        


    public:
        size_t calculateTotalSize() const {
            size_t size = 1;
            for (auto dim : dimensions) {
                size *= dim;
            }
            return size;
        }
        Matrix(const std::vector<size_t>& dims) : dimensions(dims) {
            data.resize(calculateTotalSize());
        }
        
        Matrix(const std::vector<size_t>& dims, const std::vector<T>& initialData) 
            : dimensions(dims), data(initialData) {
            if (initialData.size() != calculateTotalSize()) {
                throw std::invalid_argument("Data size doesn't match dimensions");
            }
        }
        
        void fillRandom(T min, T max) {
            std::random_device rd;
            std::mt19937 gen(rd());
            
            if constexpr (std::is_floating_point<T>::value) {
                std::uniform_real_distribution<T> dist(min, max);
                for (auto& val : data) {
                    val = dist(gen);
                }
            } else if constexpr (std::is_integral<T>::value) {
                std::uniform_int_distribution<T> dist(min, max);
                for (auto& val : data) {
                    val = dist(gen);
                }
            }
        }
        void instantiateData(std::vector<T>* startingData){
            data = std::move(startingData);
        }
        
        std::vector<T>& getVectorizedData() { return data; }
        const std::vector<T>& getVectorizedData() const { return data; }
        T* getData() { return data.data(); } // read/write
        const T* getData() const { return data.data(); } //  readonly
        
        T& at(const std::vector<size_t>& indices) {
            return data[calculateIndex(indices)];
        } // read/write
        
        const T& at(const std::vector<size_t>& indices) const {
            return data[calculateIndex(indices)];
        } // readonly
        
        const std::vector<size_t>& getDimensions() const {
            return dimensions;
        }

        template<typename Func>
        void applyFunc(Func function){
            for (auto& val : data){
                val = function(val);
            }
        }
        Matrix<T> transpose() {
            const auto& dims = getDimensions();
            if (dims.size() != 2) {
                throw std::invalid_argument("Transpose is only supported for 2D matrices.");
            }
            size_t rows = dims[0];
            size_t cols = dims[1];
            
            Matrix<T> transposed({cols, rows});
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    transposed.at({j, i}) = at({i, j});
                }
            }
            return transposed;
        }

        void print() const {
            if (dimensions.size() == 2) {
                for (size_t i = 0; i < dimensions[0]; ++i) {
                    std::cout << "[ ";
                    for (size_t j = 0; j < dimensions[1]; ++j) {
                        std::cout << data[i * dimensions[1] + j] << " ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }
    };

    
    // Cache Optimization + Parallelism 
    template<typename T>
    Matrix<T> highPerformanceMultiply(const Matrix<T>& a, const Matrix<T>& b, size_t numThreads = 0) {
        const auto& dimsA = a.getDimensions();
        const auto& dimsB = b.getDimensions();
        
        if (dimsA.size() != 2 || dimsB.size() != 2 || dimsA[1] != dimsB[0]) {
            std::cout << dimsA[0] << ", " << dimsA[1] << std::endl;
            std::cout << dimsB[0] << ", " << dimsB[1] << std::endl;
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
            
        }
        
        const size_t M = dimsA[0];  // Rows of A
        const size_t K = dimsA[1];  // Cols of A, Rows of B
        const size_t N = dimsB[1];  // Cols of B
        
        std::vector<T> resultData(M * N, 0);
        
        // Determine number of threads to use
        if (numThreads == 0) {
            numThreads = std::thread::hardware_concurrency();
        }
        numThreads = std::min(numThreads, M);  // Don't use more threads than rows
        
        std::vector<std::future<void>> futures;
        
        // Pre-transpose B for better cache access in the inner loop
        std::vector<T> B_transposed(K * N);
        const T* B_ptr = b.getData();
        
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                B_transposed[j * K + k] = B_ptr[k * N + j];
            }
        }
        
        // Calculate rows per thread
        size_t rowsPerThread = M / numThreads;
        size_t remainingRows = M % numThreads;
        
        size_t startRow = 0;
        
        // Launch worker threads
        for (size_t t = 0; t < numThreads; ++t) {
            size_t threadRows = rowsPerThread + (t < remainingRows ? 1 : 0);
            size_t endRow = startRow + threadRows;
            
            // Create and launch a worker task
            futures.push_back(std::async(std::launch::async, [&, startRow, endRow]() {
                const T* A = a.getData();
                
                // Block sizes tuned for cache
                constexpr size_t BLOCK_SIZE_M = 32;
                constexpr size_t BLOCK_SIZE_N = 128;
                constexpr size_t BLOCK_SIZE_K = 64;
                
                // Process assigned rows with cache-blocking
                for (size_t i0 = startRow; i0 < endRow; i0 += BLOCK_SIZE_M) {
                    size_t iLimit = std::min(i0 + BLOCK_SIZE_M, endRow);
                    
                    for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE_N) {
                        size_t jLimit = std::min(j0 + BLOCK_SIZE_N, N);
                        
                        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE_K) {
                            size_t kLimit = std::min(k0 + BLOCK_SIZE_K, K);
                            
                            for (size_t i = i0; i < iLimit; ++i) {
                                for (size_t j = j0; j < jLimit; ++j) {
                                    T sum = resultData[i * N + j];
                                    
                                    for (size_t k = k0; k < kLimit; ++k) {
                                        sum += A[i * K + k] * B_transposed[j * K + k];
                                    }
                                    
                                    resultData[i * N + j] = sum;
                                }
                            }
                        }
                    }
                }
            }));
            
            startRow = endRow;
        }
        
        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        return Matrix<T>({M, N}, resultData);
    }


    template<typename T> // only double or flaot
    class NeuralNetwork{
    private:
        std::vector<Matrix<T>> weights; // for mnist, would be 784x128, 128x64, 64x10
        std::vector<T> biases; // for mnist, have 3 weight layers so 
        std::vector<size_t> dimensions; // Tells us how to access biases, would be 128, 64, 10, so we can use a slider to access vals
        std::vector<T> accessed_biases; // Stores the biases we currently are accessing
        std::vector<Matrix<T>> activations;
        std::vector<Matrix<T>> gradients;
        T ReLU(T val){
            return std::max((T)0, val);
        } 
        void softmax(Matrix<T>& vals) {
            T maxVal = *std::max_element(vals.getVectorizedData().begin(), vals.getVectorizedData().end());
            vals.applyFunc([&](T val) { return exp(val / maxVal); });
            T sum = std::accumulate(vals.getVectorizedData().begin(), vals.getVectorizedData().end(), (T)0);
            vals.applyFunc([sum](T val) { return val / sum; });
        }
    public:
        NeuralNetwork(std::vector<size_t>& n) {
            // take [728, 128, 64, 10] and make multiple weight matrices, make biases
            size_t currentSum = 0;
            dimensions.resize(1);
            for(int i = 0; i < n.size() - 1; ++i){

                // bias loading
                for(int j = 0; j < n.at(i + 1); ++j){
                    biases.push_back(getRandom(-1.0f, 1.0f));
                }
                currentSum = std::accumulate(dimensions.end() - 1, dimensions.end(), 0); 
                dimensions.push_back(currentSum + n.at(i+1));

                // weights loading
                Matrix<float> weight_layer({n.at(i), n.at(i+1)});
                weight_layer.fillRandom(-1.0f, 1.0f);
                weights.push_back(weight_layer);
                for(size_t i = 0; i < dimensions.size() - 1; ++i){
                    activations.push_back(Matrix<T>({1, dimensions[i+1]})); // output of each layer
                    gradients.push_back(Matrix<T>({1, dimensions[i+1]})); // error derivs
                }

            }


        }
        void print_dimensions(){
            for(const T& dim : dimensions){
                std::cout << dim << std::endl;
            }
        }
        T Loss(const Matrix<T>& predictions, const Matrix<T>& targets){
            const std::vector<T> predicted_data = predictions.getVectorizedData();
            const std::vector<T> target_data = targets.getVectorizedData();

            T loss = 0;
            for(size_t i = 0; i < predicted_data.size(); ++i){
                loss += -target_data[i] * std::log(predicted_data[i] + 1e-9);
                std::cout << target_data[i] << ", " << predicted_data.at(i) << std::endl;
            }
            std::cout << loss << std::endl;
            return loss / predicted_data.size();
        }
        std::vector<Matrix<T>> backward(const Matrix<T>& inputs, const Matrix<T>& targets){
            activations[0] = inputs;

            Matrix<T> temp = inputs;
            for(int i = 0; i < weights.size(); ++i){
                temp = highPerformanceMultiply(temp, weights.at(i));

                accessed_biases.assign(biases.begin() + dimensions.at(i), biases.begin() + dimensions.at(i+1));
                for(int j = 0; j < temp.getVectorizedData().size(); ++j){
                    temp.getVectorizedData()[j] += accessed_biases.at(j);
                }
                if(i < weights.size() - 1){
                    temp.applyFunc([this](T val) { return this->ReLU(val);});
                } else {
                    softmax(temp);
                }

                activations[i+1] = temp;
            } // this is legit the same forward function as below but it stores each layer output for training or whatever

            Matrix<T> outputActivation = activations.back();
            std::vector<T> outputData = outputActivation.getVectorizedData();
            std::vector<T> targetData = targets.getVectorizedData();

            // error gradient yummy
            for(size_t i = 0; i < outputData.size(); ++i){
                gradients.back().getVectorizedData()[i] = outputData[i] - targetData[i];
            }

            // backpropagation stuff
            for(int i = weights.size() - 1; i > 0; --i){
                Matrix<T> previousGradient= highPerformanceMultiply(gradients[i], weights[i]);
                std::vector<T>& prevActivation = activations[i].getVectorizedData();
                for(size_t j = 0; j < previousGradient.getVectorizedData().size(); ++j){
                    previousGradient.getVectorizedData()[j] *= (prevActivation[j] > 0) ? (T) 1.0 : (T) 0.0;
                }
                gradients[i-1] = previousGradient;
            }

            return gradients;

        }
        void updateParams(const std::vector<Matrix<T>>& gradients, T learningRate){
            for(size_t i = 0; i < weights.size(); ++i){
                for(size_t j = 0; j < weights[i].getVectorizedData().size(); ++j){
                    weights[i].getVectorizedData()[j] -= learningRate * gradients[i].getVectorizedData()[j];
                }

                size_t start = dimensions.at(i);
                size_t end = dimensions.at(i + 1);
                for(size_t j = start; j < end; ++j){
                    biases[j] -= learningRate * gradients.at(i).getVectorizedData()[j-start];
                }
            }
        }
        Matrix<T> forward(Matrix<T> inputs) {
            Matrix<T> temp = inputs;
            for(int i = 0; i < weights.size(); ++i){

                temp = highPerformanceMultiply(temp, weights.at(i));
                
                // in order to add biases, yoink a piece of the vector
                accessed_biases.assign(biases.begin() + dimensions.at(i), biases.begin() + dimensions.at(i+1));
                for(int i = 0; i < temp.getVectorizedData().size(); i++){
                    temp.getVectorizedData()[i] += accessed_biases.at(i);
                }
                if(i < weights.size() - 1){
                    temp.applyFunc([this](T val) { return this->ReLU(val);});
                } else {
                    softmax(temp);
                }

            }

            return temp;
        }

    };

    class MNISTDataLoader {
    private:
        std::vector<std::vector<float>> images;
        std::vector<int> labels;

    public:
        // Load images and labels from a combined CSV file
        void loadFromCombinedCSV(const std::string& combinedFile) {
            // Clear existing data
            images.clear();
            labels.clear();

            // Open the combined CSV file
            std::ifstream file(combinedFile);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open combined CSV file: " + combinedFile);
            }

            std::string line;
            while (std::getline(file, line)) {
                std::vector<float> image;
                std::istringstream ss(line);
                std::string value;

                // First value is the label
                std::getline(ss, value, ',');
                int label = std::stoi(value);
                labels.push_back(label);

                // Rest of the values are pixels
                while (std::getline(ss, value, ',')) {
                    // Normalize pixel values to [0,1]
                    image.push_back(std::stof(value) / 255.0f);
                }

                // Ensure image is 1x784 (28x28 flattened)
                if (image.size() != 784) {
                    throw std::runtime_error("Invalid image size. Expected 784 pixels. Got: " + 
                                            std::to_string(image.size()));
                }

                images.push_back(image);
            }
        }

        // Get total number of images
        size_t getImageCount() const {
            return images.size();
        }

        // Access a specific image
        const std::vector<float>& getImage(size_t index) const {
            if (index >= images.size()) {
                throw std::out_of_range("Image index out of range");
            }
            return images[index];
        }

        // Access a specific label
        int getLabel(size_t index) const {
            if (index >= labels.size()) {
                throw std::out_of_range("Label index out of range");
            }
            return labels[index];
        }

        // Batch retrieval for training
        std::pair<std::vector<std::vector<float>>, std::vector<int>> 
        getBatch(size_t startIndex, size_t batchSize) const {
            std::vector<std::vector<float>> batchImages;
            std::vector<int> batchLabels;

            for (size_t i = startIndex; i < startIndex + batchSize && i < images.size(); ++i) {
                batchImages.push_back(images[i]);
                batchLabels.push_back(labels[i]);
            }
            
            return {batchImages, batchLabels};
        }

        // Utility method to print dataset statistics
        void printDatasetInfo() const {
            std::cout << "Dataset Statistics:" << std::endl;
            std::cout << "Total Images: " << images.size() << std::endl;
            
            // Count labels
            std::vector<int> labelCounts(10, 0);
            for (int label : labels) {
                labelCounts[label]++;
            }

            std::cout << "Label Distribution:" << std::endl;
            for (int i = 0; i < 10; ++i) {
                std::cout << "Digit " << i << ": " << labelCounts[i] << " images" << std::endl;
            }
        }
    };

    // Benchmark utility
    template<typename MatrixFunc>
    double benchmark(MatrixFunc func, size_t iterations = 5) {
        std::vector<double> times;
        
        for (size_t i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            times.push_back(elapsed.count());
        }
        
        // Return average time (excluding the first run as warmup)
        double sum = 0.0;
        for (size_t i = 1; i < times.size(); ++i) {
            sum += times[i];
        }
        return sum / (times.size() - 1);
    }

    // Test and benchmark all methods
    void runBenchmarks() {
        std::vector<size_t> input_size = {784, 128, 64, 10};
        NeuralNetwork<float> nn(input_size);
        Matrix<float> inputs({1, 784});

        double highPerfTime = benchmark([&]() { 
            
            inputs.fillRandom(0.0f, 1.0f);
            

        }, 10); 
        // about 2.24 seconds for 1k reps for forward passes ONLY 


        

        std::cout << "Combined high-performance: " << highPerfTime <<  std::endl;
        
    }


        
    int main() {

        std::vector<size_t> networkArchitecture = {784, 128, 10};
        NeuralNetwork<float> nn(networkArchitecture);
        MNISTDataLoader loader;
        try {
            loader.loadFromCombinedCSV("mnist_test.csv");
            
            size_t batchSize = 32;
            size_t numBatches = 10;  // Calculate number of batches based on dataset size and batch size
            
            size_t epochs = 10;  // Define how many epochs to train
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                for (size_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
                    auto [batch, batchLabels] = loader.getBatch(batchIndex, batchSize);
    
                    for (size_t i = 0; i < batch.size(); ++i) {
                        Matrix<float> inputImage({1, 784});
                        for (size_t j = 0; j < networkArchitecture[0]; ++j) {
                            inputImage.getVectorizedData()[j] = batch[i][j];  // Normalize pixel values
                        }
    
                        Matrix<float> labelMatrix({1, 10}, std::vector<float>(10, 0.0f));
                        int labelValue = batchLabels[i];
                        labelMatrix.getVectorizedData()[labelValue] = 1.0f;
                        
                        auto vectorizedLabel = labelMatrix.getVectorizedData();
                        auto gradients = nn.backward(inputImage, labelMatrix);

                        std::cout << inputImage.calculateTotalSize() << std::endl; 
                        for(const auto& val : inputImage.getVectorizedData()){
                            std::cout << val << "\n";
                        }
                        float loss = nn.Loss(gradients.front(), labelMatrix);
                        std::cout << "Epoch " << epoch + 1 << ", Batch " << batchIndex + 1
                                  << ", Loss for image " << i << " (Label " << labelValue << "): " << loss << std::endl;
                        std::cout << "Guessed: ";
                        for(const auto& val : nn.forward(inputImage).getVectorizedData()){
                            std::cout << val << ", ";
                        }
                        std::cout << "\n";
                        nn.updateParams(gradients, 0.01f);  // Update weights
                    }
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    
        return 0;
    }
    
