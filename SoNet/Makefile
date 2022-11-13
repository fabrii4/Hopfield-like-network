##############################
# SoNet Makefile
##############################

#INSTALLROOT=$(PWD)

CC=gcc
CPP=g++
INSTALL=install
BINARY=SoNet
#BIN=/usr/local/bin
#VERSION = 0.0.1

#LIBX11FLAGS= -I/usr/X11R6/include -L/usr/X11R6/lib
#SDLLIBS = $(shell sdl-config --libs) -lturbojpeg -lSDL_gfx 
#SDLFLAGS = $(shell sdl-config --cflags)
#CFLAGS += -DUSE_SDL -O2 -DLINUX -DVERSION=\"$(VERSION)\" -I$(SDLFLAGS) $(WARNINGS)
#CPPFLAGS = $(CFLAGS)
OPENCVLIBS = -I /usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

#COMMON_OBJECTS=
#TARGET_OBJECTS= SoNet.o

SRC_DIR= ./
OBJ_DIR= ./obj
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
		
all:	SoNet
clean:
	@echo "Cleaning up obj directory."
	rm -f -r $(OBJ_DIR)
	rm -f *.a *.o $(BINARY)

# Applications:
#SoNet:	$(COMMON_OBJECTS) $(TARGET_DIR)/$(TARGET_OBJECTS)
#	$(CPP) $(COMMON_OBJECTS) $(TARGET_DIR)/$(TARGET_OBJECTS) $(OPENCVLIBS) -o $(BINARY)
#	chmod 755 $(BINARY)

SoNet: $(OBJ_FILES)
	$(CPP) -o $@ $^ $(OPENCVLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CPP) -c -o $@ $< $(OPENCVLIBS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)


#install: SoNet
#	$(INSTALL) -s -m 755 -g root -o root $(BINARY) $(BIN) 
#	rm -f $(BIN)/$(BINARY)
