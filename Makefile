CC = g++
CFLAGS = -Wall -Werror -std=c++17

TEST_DIR = tests
BIN_DIR = bin
SRC_DIR = src

TEST_EXECUTABLE = $(BIN_DIR)/run_tests

TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_LDFLAGS = 

.PHONY: test build_test clean_test

test: build_test
	@echo "--- EXECUTING TESTS ---"
	@$(TEST_EXECUTABLE)
	@echo "--- FINISHED ---"

build_test: $(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_SOURCES)
	@mkdir -p $(BIN_DIR)
	@echo "Compiling tests..."
	$(CC) $(CFLAGS) $^ -o $@ $(TEST_LDFLAGS)
	@echo "Test executable created at: $(TEST_EXECUTABLE)"

clean_test:
	@echo "cleaning tests..."
	@rm -f $(TEST_EXECUTABLE)