Write a class `Emitter`, a small event emitter.

- `on(event, fn)` registers `fn` for `event`; `once(event, fn)` registers it so that it runs for the next emit only and is then removed. Both return the emitter (so calls can be chained). Event names can be any value. Registering the same function twice registers it twice. Both throw a `TypeError` if `fn` is not a function.
- `off(event, fn)` removes every registration of `fn` for `event`, including ones made with `once`, and returns the emitter. Removing something that is not registered does nothing.
- `emit(event, ...args)` calls the listeners of `event` in the order they were registered, with `this` set to the emitter and `args` as arguments. It returns `true` if at least one listener was called and `false` otherwise. Listeners added or removed while an emit is running do not change which listeners run in that emit (the list is fixed when `emit` starts); a `once` listener is removed before it is called, so an emit triggered from inside it does not call it again.
- `listenerCount(event)` returns the number of registrations for `event`.
