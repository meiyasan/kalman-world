CXX     := g++
CXXFLAGS := $(shell root-config --cflags) -O2
LDFLAGS  := $(shell root-config --libs)
TARGET   := violin_fit

.PHONY: all run clean

all: $(TARGET)

$(TARGET): main.C
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) fits_out.root \
	    mode_1_individual_fits.pdf mode_2_individual_fits.pdf mode_3_individual_fits.pdf \
	    global_fits_all_modes.pdf
