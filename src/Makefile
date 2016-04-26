CPPFLAGS = -I /usr/include

random_forest-1: random_forest-1.cc information_gain.o random_forest.o
	clang++ -std=gnu++11 $(CPPFLAGS) -o $@ $<

%.o: %.cc
	clang++ -std=gnu++11 $(CPPFLAGS) -c -o $@ $<

clean:
	rm -f random_forest-1 *.o
