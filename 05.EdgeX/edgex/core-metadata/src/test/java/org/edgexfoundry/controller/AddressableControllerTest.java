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
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import static org.edgexfoundry.test.data.AddressableData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.controller.impl.AddressableControllerImpl;
import org.edgexfoundry.controller.impl.CallbackExecutor;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.AddressableData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class AddressableControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private AddressableControllerImpl controller;

  @Mock
  private AddressableRepository repos;

  @Mock
  private AddressableDao dao;

  @Mock
  private CallbackExecutor callback;

  private Addressable addr;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    addr = newTestInstance();
    addr.setId(TEST_ID);
  }

  @Test
  public void testAddressable() {
    when(repos.findOne(TEST_ID)).thenReturn(addr);
    assertEquals("Addressable returned is not as expected", addr, controller.addressable(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testAddressableNotFound() {
    controller.addressable(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testAddressableException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressable(TEST_ID);
  }

  @Test
  public void testAddressables() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.findAll(any(Sort.class))).thenReturn(addrs);
    when(repos.count()).thenReturn(1L);
    List<Addressable> addressables = controller.addressables();
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertEquals("Addressable returned is not as expected", addr, addressables.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testAddressablesMaxLimit() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.count()).thenReturn(1000L);
    controller.addressables();
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesException() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.addressables();
  }

  @Test
  public void testAddressableForName() {
    when(repos.findByName(AddressableData.TEST_ADDR_NAME)).thenReturn(addr);
    assertEquals("Addressable returned is not as expected", addr,
        controller.addressableForName(AddressableData.TEST_ADDR_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testAddressableForNameNotFound() {
    controller.addressableForName(AddressableData.TEST_ADDR_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testAddressableForNameException() {
    when(repos.findByName(AddressableData.TEST_ADDR_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressableForName(AddressableData.TEST_ADDR_NAME);
  }

  @Test
  public void testAddressableByAddress() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.findByAddress(AddressableData.TEST_ADDRESS)).thenReturn(addrs);
    List<Addressable> addressables = controller.addressablesByAddress(AddressableData.TEST_ADDRESS);
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertEquals("Addressable returned is not as expected", addr, addressables.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testAddressableByAddressException() {
    when(repos.findByAddress(AddressableData.TEST_ADDRESS))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressablesByAddress(AddressableData.TEST_ADDRESS);
  }

  @Test
  public void testAddressableByPort() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.findByPort(AddressableData.TEST_PORT)).thenReturn(addrs);
    List<Addressable> addressables = controller.addressablesByPort(AddressableData.TEST_PORT);
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertEquals("Addressable returned is not as expected", addr, addressables.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testAddressableByPortException() {
    when(repos.findByPort(AddressableData.TEST_PORT)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressablesByPort(AddressableData.TEST_PORT);
  }


  @Test
  public void testAddressableByTopic() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.findByTopic(AddressableData.TEST_TOPIC)).thenReturn(addrs);
    List<Addressable> addressables = controller.addressablesByTopic(AddressableData.TEST_TOPIC);
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertEquals("Addressable returned is not as expected", addr, addressables.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testAddressableByTopicException() {
    when(repos.findByTopic(AddressableData.TEST_TOPIC))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressablesByTopic(AddressableData.TEST_TOPIC);
  }

  @Test
  public void testAddressableByPublisher() {
    List<Addressable> addrs = new ArrayList<>();
    addrs.add(addr);
    when(repos.findByPublisher(AddressableData.TEST_PUBLISHER)).thenReturn(addrs);
    List<Addressable> addressables =
        controller.addressablesByPublisher(AddressableData.TEST_PUBLISHER);
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertEquals("Addressable returned is not as expected", addr, addressables.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testAddressableByPublisherException() {
    when(repos.findByPublisher(AddressableData.TEST_PUBLISHER))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressablesByPublisher(AddressableData.TEST_PUBLISHER);
  }

  @Test
  public void testAdd() {
    when(repos.save(addr)).thenReturn(addr);
    assertEquals("Addressable ID returned is not the value expected", TEST_ID,
        controller.add(addr));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddDuplicateKey() {
    when(repos.save(addr)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    controller.add(addr);
  }

  @Test(expected = ServiceException.class)
  public void testAddServiceException() {
    when(repos.save(addr)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(addr);
  }

  @Test
  public void testUpdate() {
    when(dao.getByIdOrName(addr)).thenReturn(addr);
    assertTrue("Addressable was not updated", controller.update(addr));
  }
  
  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateNotFoundException() {
    when(dao.getByIdOrName(addr)).thenReturn(null);
    controller.update(addr);
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateWithAssociatedDevices() {
    Addressable addr2 = AddressableData.newTestInstance();
    addr2.setName("foo");
    when(dao.getByIdOrName(addr)).thenReturn(addr2);
    when(dao.isAddressableAssociatedToDevice(addr2)).thenReturn(true);
    controller.update(addr);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(addr);
    assertTrue("Addressable was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteDaoFails() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.delete(TEST_ID);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWithAssociatedDevices() {
    Addressable addr2 = AddressableData.newTestInstance();
    addr2.setName("foo");
    when(repos.findOne(TEST_ID)).thenReturn(addr2);
    when(dao.isAddressableAssociatedToDevice(addr2)).thenReturn(true);
    controller.delete(TEST_ID);
  }

  @Test
  public void testDeleteByName() {
    when(repos.findByName(AddressableData.TEST_ADDR_NAME)).thenReturn(addr);
    assertTrue("Addressable was not deleted",
        controller.deleteByName(AddressableData.TEST_ADDR_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(AddressableData.TEST_ADDR_NAME)).thenReturn(null);
    controller.deleteByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameDaoFails() {
    when(repos.findByName(AddressableData.TEST_ADDR_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deleteByName(AddressableData.TEST_ADDR_NAME);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }

}
