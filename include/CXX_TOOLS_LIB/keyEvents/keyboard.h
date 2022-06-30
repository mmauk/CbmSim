#ifndef KEYBOARD_H_
#define KEYBOARD_H_

#include <iostream>
#include <fcntl.h>
#include <pthread.h>
#include <linux/input.h>
 
#define KEYBOARD_DEV "/dev/input/event0"

/*
 * structure representing the states that our
 * keys can be in
 */
struct keyboard_state
{
    signed short keys[KEY_CNT];
};
 
class cKeyboard
{
  public:
    cKeyboard();
    ~cKeyboard();

	/*
	 * main keyboard event loop. obj is cast to
	 * type cKeyboard
	 *
	 */
    static void* loop(void* obj);

	/*
	 * reads the last event registered within the event loop,
	 * updating the keyboard state with the new status of the
	 * key associated with the given event
	 *
	 */
    void readEv();

	/*
	 * a getter for a given keys state.
	 *
	 */
    short getKeyState(short key);

  protected:
  private:
    pthread_t thread;
    bool active;
    int keyboard_fd;
    input_event *keyboard_ev;
    keyboard_state *keyboard_st;
    char name[256];
};

#endif /* KEYBOARD_H_ */

