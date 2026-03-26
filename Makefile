NVCC = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fopenmp -O3
TARGET = a2
SRCS = submission.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET) knn.txt approx_knn.txt kmeans.txt

# Package for submission: creates a2.zip with folder a2/ inside
submit: clean
	@mkdir -p /tmp/a2_pkg/a2
	@cp Makefile $(SRCS) /tmp/a2_pkg/a2/
	@[ -f report.pdf ] && cp report.pdf /tmp/a2_pkg/a2/ || echo "WARNING: report.pdf not found - add it before final submission"
	@cd /tmp/a2_pkg && zip -r "$(CURDIR)/a2.zip" a2
	@rm -rf /tmp/a2_pkg
	@echo "Created a2.zip"

.PHONY: all clean submit
