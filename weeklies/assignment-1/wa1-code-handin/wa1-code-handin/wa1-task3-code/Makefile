CXX        = nvcc -O3 -arch=sm_80

SRC        = wa1-task3.cu
EXECUTABLE = wa1-task3

default: compile run

.cu.o: $(SRC)
	$(CXX) -c -o $@ $<

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) helper.h
	$(CXX) -o $(EXECUTABLE) $(SRC)

run: $(EXECUTABLE)
	./$(EXECUTABLE) 753411

clean:
	rm -f $(EXECUTABLE) *.o
