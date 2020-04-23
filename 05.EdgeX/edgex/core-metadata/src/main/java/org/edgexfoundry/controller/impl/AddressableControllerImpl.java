/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller.impl;

import java.util.List;

import org.edgexfoundry.controller.Action;
import org.edgexfoundry.controller.AddressableController;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/addressable")
public class AddressableControllerImpl implements AddressableController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(AddressableControllerImpl.class);

  private static final String ERR_MSG =
      "is still referenced by existing devices or device services.";

  private static final String ERR_GET = "Error getting addressables:  ";

  @Autowired
  private AddressableRepository repos;

  @Autowired
  private AddressableDao dao;

  @Autowired
  private CallbackExecutor callback;
  
  @Value("${read.max.limit:100}")
  private int maxLimit;

  /**
   * Fetch a specific addressable by database generated id. May return null if no addressable
   * matches on id. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if not found by id.
   * 
   * @param String addressable database generated id
   * @return Addressable
   */
  @Override
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  public Addressable addressable(@PathVariable String id) {
    try {
      Addressable addr = repos.findOne(id);
      if (addr == null) {
        throw new NotFoundException(Addressable.class.toString(), id);
      }
      return addr;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting addressable:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all addressable objects sorted by database generated id. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns LimitExceededException (HTTP 413) if the
   * number returned exceeds the max limit.
   * 
   * @return list of addressable
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<Addressable> addressables() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded requesting addressables");
        throw new LimitExceededException("Addressable");
      } else {
        Sort sort = new Sort(Sort.Direction.DESC, "_id");
        return repos.findAll(sort);
      }
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Addressable with matching name (name should be unique). May be null if none match.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if not found by name.
   * 
   * @param name
   * @return addressable matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public Addressable addressableForName(@PathVariable String name) {
    try {
      Addressable addr = repos.findByName(name);
      if (addr == null)
        throw new NotFoundException(Addressable.class.toString(), name);
      return addr;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting addressable:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Addressable objects with given address. List may be empty if none are associated to the
   * address. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param address string (like a URL address)
   * @return list of addressable matching address
   */
  @RequestMapping(value = "/address/{address:.+}", method = RequestMethod.GET)
  @Override
  public List<Addressable> addressablesByAddress(@PathVariable String address) {
    try {
      return repos.findByAddress(address);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Addressable objects with given port. List may be empty if none are associated to the
   * port. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param port
   * @return list of addressable matching port
   */
  @RequestMapping(value = "/port/{port}", method = RequestMethod.GET)
  @Override
  public List<Addressable> addressablesByPort(@PathVariable int port) {
    try {
      return repos.findByPort(port);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Addressable objects with given topic. List may be empty if none are associated to the
   * topic. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param topic
   * @return list of addressable matching topic
   */
  @RequestMapping(value = "/topic/{topic:.+}", method = RequestMethod.GET)
  @Override
  public List<Addressable> addressablesByTopic(@PathVariable String topic) {
    try {
      return repos.findByTopic(topic);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Addressable objects with given publisher. List may be empty if none are associated to
   * the publisher. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param publisher
   * @return list of addressable matching publisher
   */
  @RequestMapping(value = "/publisher/{publisher:.+}", method = RequestMethod.GET)
  @Override
  public List<Addressable> addressablesByPublisher(@PathVariable String publisher) {
    try {
      return repos.findByPublisher(publisher);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new Addressable - name must be unique. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues.
   * 
   * @param Addressable to add
   * @return new database generated id for the new Addressable
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody Addressable addressable) {
    if (addressable == null)
      throw new ServiceException(new DataValidationException("No addressable data provided"));
    try {
      repos.save(addressable);
      return addressable.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + addressable.getName());
    } catch (Exception e) {
      logger.error("Error adding Addressable:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the Addressable identified by the id or name in the object provided. Id is used first,
   * name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no addressable with
   * the provided id is found.
   * 
   * @param Addressable object holding the identifier and new values for the Addressable
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody Addressable addressable2) {
    if (addressable2 == null)
      throw new ServiceException(new DataValidationException("No addressable data provided"));
    try {
      Addressable addressable = dao.getByIdOrName(addressable2);
      if (addressable == null) {
        logger.error("Request to update with non-existent or unidentified addressable (id/name):  "
            + addressable2.getId() + "/" + addressable2.getName());
        throw new NotFoundException(Addressable.class.toString(), addressable2.getId());
      }
      updateAddressable(addressable2, addressable);
      notifyAssociates(addressable, Action.PUT);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error updating addressable:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateAddressable(Addressable from, Addressable to) {
    if (from.getProtocol() != null)
      to.setProtocol(from.getProtocol());
    if (from.getAddress() != null)
      to.setAddress(from.getAddress());
    if (from.getMethod() != null)
      to.setMethod(from.getMethod());
    if (from.getPath() != null)
      to.setPath(from.getPath());
    if (from.getPort() != 0)
      to.setPort(from.getPort());
    if (from.getPublisher() != null)
      to.setPublisher(from.getPublisher());
    if (from.getTopic() != null)
      to.setTopic(from.getTopic());
    if (from.getUser() != null)
      to.setUser(from.getUser());
    if (from.getPassword() != null)
      to.setPassword(from.getPassword());
    if (from.getName() != null && !from.getName().equals(to.getName())) {
      checkAddressableAssociatedToDevice(to, from.getName());
    }
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
  }
  
  private void checkAddressableAssociatedToDevice(Addressable addressable, String oldName) {
    if ((dao.isAddressableAssociatedToDevice(addressable)
        || dao.isAddressableAssociatedToDeviceService(addressable))) {
      logger.error("Data integrity issue. Addressable with name: " + oldName + ERR_MSG);
      throw new DataValidationException(
          "Data integrity issue. Addressable with name: " + oldName + ERR_MSG);
    } else
      addressable.setName(oldName);
  }

  /**
   * Remove the Addressable designated by the database generated id for the Addressable. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no addressable with the provided id is found.
   * 
   * @param database generated id
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      Addressable addressable = repos.findOne(id);
      if (addressable == null) {
        logger.error("Request to delete with non-existent addressable id:  " + id);
        throw new NotFoundException(Addressable.class.toString(), id);
      }
      return deleteAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing addressable:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the Addressable designated by unique name identifier. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * addressable with the provided name is found.
   * 
   * @param unique name of the Addressable
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      Addressable addressable = repos.findByName(name);
      if (addressable == null) {
        logger.error("Request to delete with unknown addressable name:  " + name);
        throw new NotFoundException(Addressable.class.toString(), name);
      }
      return deleteAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing addressable:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean deleteAddressable(Addressable addressable) {
    if (dao.isAddressableAssociatedToDevice(addressable)
        || (dao.isAddressableAssociatedToDeviceService(addressable))) {
      logger.error("Data integrity issue. Addressable with id: " + addressable.getId() + ERR_MSG);
      throw new DataValidationException(
          "Data integrity issue. Addressable with id: " + addressable.getId() + ERR_MSG);
    }
    repos.delete(addressable);
    return true;
  }

  private void notifyAssociates(Addressable addressable, Action action) {
    callback.callback(dao.getOwningServices(addressable), addressable.getId(), action,
        ActionType.ADDRESSABLE);
  }

}
