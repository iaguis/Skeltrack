#include <glib.h>

struct _PQueue_element {
  gpointer data;
  guint priority;
};

typedef struct _PQueue_element PQelement;

struct _PQueue {
  PQelement *elements;
  GHashTable *map;
  guint size;
  guint max_size;
};

typedef struct _PQueue PQueue;

PQueue *        pqueue_new                      (guint           max_size);

void            pqueue_insert                   (PQueue         *pqueue,
                                                 gpointer        data,
                                                 guint           priority);

gpointer        pqueue_pop_minimum              (PQueue         *pqueue);

void            pqueue_delete                   (PQueue         *pqueue,
                                                 gpointer        data);

gboolean        pqueue_has_element              (PQueue         *pqueue,
                                                 gpointer        data);

gboolean        pqueue_is_empty                 (PQueue         *pqueue);

void            pqueue_free                     (PQueue         *pqueue);
