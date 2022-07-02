/*
 *  FILE: tty.h
 *	AUTHOR: Sean Gallogly
 *  CREATED: 07/01/2022
 */

#ifndef TTY_MANIP_
#define TTY_MANIP_

#include <stdio.h>

/*
 * DESCRIP: sets the tty pointed to by fp to noncanonical mode,
 * 			and stops the tty from echoing user input
 *
 * NOTE: assumes fp points to an open tty, so ensure it does before calling.
 */
void enable_raw_mode(FILE **fp);

/*
 * DESCRIP: sets the tty pointed to by fp to canonical mode,
 * 			and returns the tty to echoing user input
 *
 * NOTES: like enable_raw_mode above, assumes fp points to an open tty.
 */
void disable_raw_mode(FILE **fp);

/*
 * DESCRIP: checks whether the user entered input from the keyboard into the tty
 * 			pointed to by fp. Does not check whether fp is valid (TODO)
 *
 * RETURNS: true if keyboard was hit, else false
 */
bool kbhit(FILE **fp);

/*
 * DESCRIP: obtains one character from the tty associated with fp. Again, does not
 * 			check whether fp is valid.
 *
 * RETURNS: the first available character in tty pointed to by fp
 */
char getchar(FILE **fp);

/*
 * DESCRIP: initialize the tty in raw mode
 *
 * RETURNS: non-negative value if able to open tty, else -1
 */
int init_tty(FILE **fp);

/*
 * DESCRIP: processes input for the tty pointed to by fp. respons to 'p' for 'pause'
 * 			and 'c' for continue, if paused already.
 */
void process_input(FILE **fp, int tts, int trial_num);

/*
 * DESCRIP: disables raw mode on the tty and closes the fp. resets fp to NULL.
 *
 * RETURNS: 0 on successfully completing the above tasks. TODO: need to check for failure.
 */
int reset_tty(FILE **fp);

#endif /* TTY_MANIP_ */

