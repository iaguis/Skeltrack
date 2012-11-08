#include <glib.h>
#include "skeltrack-util.h"

struct _PQueue_element {
  Node  *data;
  guint priority;
};

typedef struct _PQueue_element PQelement;

struct _PQueue {
  PQelement *elements;
  guint *map;
  guint size;
  guint max_size;
  guint width;
  guint height;
};

typedef struct _PQueue PQueue;

PQueue *        pqueue_new                      (guint           max_size,
                                                 guint           width,
                                                 guint           height);

void            pqueue_insert                   (PQueue         *pqueue,
                                                 Node           *data,
                                                 guint           priority);

Node *          pqueue_pop_minimum              (PQueue         *pqueue);

void            pqueue_delete                   (PQueue         *pqueue,
                                                 Node           *data);

gboolean        pqueue_has_element              (PQueue         *pqueue,
                                                 Node           *data);

gboolean        pqueue_is_empty                 (PQueue         *pqueue);

void            pqueue_free                     (PQueue         *pqueue);
